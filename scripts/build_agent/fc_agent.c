/* fc_agent.c  - static helper for Firecracker Gym executors
 *
 * Protocol (all 32-bit little-endian integers):
 *   timeout      - seconds (<=0  : no timeout)
 *   use_shell    - 0/1
 *   max_output   - 0 : unlimited, >0 : keep that many bytes
 *   cmd_len      - length of following UTF-8 command string
 *   <bytes>      - the command itself (NOT NUL terminated)
 *
 * Response:
 *   exit_code    - int32
 *   output_len   - int32
 *   <bytes>      - (min(output_len,max_output)) bytes of captured output
 *
 * The agent always drains the child’s pipe completely so the child can exit,
 * but only returns up to max_output bytes in RAM.
 */
#define _GNU_SOURCE
#include <sys/socket.h>
#include <linux/vm_sockets.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/reboot.h>
#include <linux/reboot.h>
#include <signal.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#ifndef FC_AGENT_PORT
#define FC_AGENT_PORT 52
#endif

#define BUF_SZ 1 << 26 // 64 mb max if no max supplied
#define MAX_CMD_LEN (1 << 18)  // 256 KiB safety cap for incoming command

static volatile sig_atomic_t timed_out = 0;      // 0=no, 1=term sent, 2=killed
static volatile sig_atomic_t child_pid_global = -1;

static ssize_t read_exact(int fd, void *buf, size_t n) {
    size_t got = 0; char *p = (char*)buf;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r == 0) return got;      // EOF
        if (r < 0) { if (errno == EINTR) continue; return r; }
        got += (size_t)r;
    }
    return (ssize_t)got;
}

static void on_alarm(int sig) {
    (void)sig;
    pid_t c = child_pid_global;
    if (timed_out == 0) {
        timed_out = 1;
        if (c > 0) kill(c, SIGTERM);   // let shell traps run briefly
        alarm(2);                       // escalate soon
    } else {
        timed_out = 2;
        if (c > 0) kill(c, SIGKILL);   // hard kill if still around
        // no further alarms
    }
}


static void serial_write(const char *s){
    int fd = open("/dev/ttyS0", O_WRONLY | O_NOCTTY);
    if (fd >= 0){ write(fd, s, strlen(s)); close(fd);}
}

static void poweroff_now(){
    sync();
    reboot(LINUX_REBOOT_CMD_POWER_OFF);
    system("poweroff -f >/dev/null 2>&1 || reboot -f >/dev/null 2>&1 || echo o > /proc/sysrq-trigger");
}

int main(){
    int s = socket(AF_VSOCK, SOCK_STREAM, 0);
    if(s<0){serial_write("FC_AGENT_ERR:socket\n"); poweroff_now(); return 1;}
    struct sockaddr_vm sa = {0};
    sa.svm_family = AF_VSOCK;
    sa.svm_cid = VMADDR_CID_ANY;
    sa.svm_port = FC_AGENT_PORT;
    if(bind(s,(struct sockaddr*)&sa,sizeof(sa))<0){serial_write("FC_AGENT_ERR:bind\n"); poweroff_now();return 1;}
    if(listen(s,1)<0){serial_write("FC_AGENT_ERR:listen\n"); poweroff_now(); return 1;}
    signal(SIGALRM, on_alarm);


    // Only print READY marker AFTER bind/listen succeeded
    serial_write("FC_AGENT_READY\n");

    while (1) {
        int c = accept(s,NULL,NULL);
        if(c<0){serial_write("FC_AGENT_ERR:accept\n"); continue;}
        int flags = fcntl(c, F_GETFD);
        fcntl(c, F_SETFD, flags | FD_CLOEXEC);

        int32_t timeout = 0, use_shell = 1, max_output_len = 0, cmd_len = 0;
        if (read_exact(c, &timeout, 4) != 4 ||
            read_exact(c, &use_shell, 4) != 4 ||
            read_exact(c, &max_output_len, 4) != 4 ||
            read_exact(c, &cmd_len, 4) != 4 ||
            cmd_len <= 0 || cmd_len > MAX_CMD_LEN) {
            close(c); continue;
        }

        char *cmd = (char*)malloc((size_t)cmd_len + 1);
        if (!cmd) { close(c); continue; }
        if (read_exact(c, cmd, (size_t)cmd_len) != cmd_len) {
            free(cmd); close(c); continue;
        }
        cmd[cmd_len] = '\0';

        // Special shutdown command: __AGENT_EXIT__
        if (strcmp(cmd, "__AGENT_EXIT__") == 0) {
            int32_t exit_code = 0, output_len = 0;
            write(c, &exit_code, sizeof(exit_code));
            write(c, &output_len, sizeof(output_len));
            close(c);
            free(cmd);
            break; // exit main loop, agent will power off below
        }

        /* allocate a buffer big enough for the user-requested cap (+NUL)          */
        /*  - if max_output_len==0 fall back to BUF_SZ (64 MiB) */
        size_t cap = (max_output_len > 0 ? (size_t)max_output_len : BUF_SZ - 1) + 1;
        char *output = malloc(cap);
        if (!output) { close(c); continue; }

        int pipefd[2]; pipe(pipefd);
        pid_t pid = fork();
        child_pid_global = pid;
        if (timeout > 0) alarm(timeout);
        if(pid == 0){
            close(pipefd[0]);
            dup2(pipefd[1], 1); dup2(pipefd[1], 2); close(pipefd[1]);
            /* make stdin a harmless /dev/null */
            int devnull = open("/dev/null", O_RDONLY);
            if (devnull >= 0){ dup2(devnull, 0); close(devnull); }
            if(use_shell)
                execl("/bin/sh", "sh", "-lc", cmd, (char*)NULL);
            else
                execl("/bin/sh", "sh", "-c", cmd, (char*)NULL);
            _exit(127);
        }
        int status = 0;
        close(pipefd[1]);
        if (timeout > 0) alarm(timeout);

        /* Drain output */
        char tmp[512];
        ssize_t r; size_t total = 0;

        while ((r = read(pipefd[0], tmp, sizeof tmp)) > 0) {
            /* copy up to cap-1 bytes into the outgoing buffer */
            if (total < cap - 1) {
                size_t to_copy = r;
                if (total + to_copy >= cap - 1)
                    to_copy = cap - 1 - total;
                memcpy(output + total, tmp, to_copy);
                total += to_copy;
            }
            /* stop the command early if we have reached the user cap */
            if (max_output_len > 0 && total >= (size_t)max_output_len) {
                kill(pid, SIGTERM);
                int waited = 0;
                while (waitpid(pid, NULL, WNOHANG) == 0 && waited < 20) {  // ~2s total
                    struct timespec ts = {.tv_sec=0, .tv_nsec=100000000};
                    nanosleep(&ts, NULL);
                    waited++;
                }
            if (waitpid(pid, NULL, WNOHANG) == 0) kill(pid, SIGKILL);
                break;
            }
        }
        output[total] = '\0';

        int rc = 0;
        pid_t w = waitpid(pid, &status, 0);
        if (w < 0) rc = 128;                 // wait error
        else if (timed_out) rc = 124;        // standard "timeout" code
        else if (WIFEXITED(status)) rc = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) rc = 128 + WTERMSIG(status);

        /* reply header is small; write it no matter what */
        int32_t exit_code = rc;
        int32_t outlen    = (int32_t)total;   // binary-safe: include NULs
        write(c, &exit_code, sizeof(exit_code));
        write(c, &outlen,    sizeof(outlen));
        if (outlen > 0) write(c, output, outlen);

        free(output);

        close(c);
        free(cmd);
    }

    close(s);
    poweroff_now();
    return 0;
}
