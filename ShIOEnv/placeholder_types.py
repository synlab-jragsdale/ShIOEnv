LOCAL_RANDOM_PLACEHOLDERS = ["[File]",  "[Directory]",  "[Executable]", "[TarFile]", "[ZipFile]"]
RANDOM_PLACEHOLDERS = ["[GlobalFile]", "[GlobalDirectory]", "[GlobalExecutable]", "[Interface]", "[Username]", "[Groupname]"]
RANDOM_STR_PLACEHOLDERS = {"[Password]": ["password", "pass", "1234", "admin"],
                           "[Word]": ["matrix", "region", "rusted", "monkey", "browse", "uproot", "outlet",
                                        "accept", "immune", "forked", "banana", "bob", "bumblebee", "ubuntu"],
                           "[String]": ["matrix", "region", "rusted", "monkey", "browse", "uproot", "outlet",
                                        "accept", "immune", "forked", "banana", "bob", "bumblebee", "ubuntu"],
                           "[NewFile]": ["file", "file.1", "file.2", "file.txt", "newfile", "newfile1"],
                           "[NewDirectory]": ["dir", "dir1", "docs", "newdir", "newdir1"],
                           "[NewExecutable]": ["exec", "newexec", "exec.sh", "esec.py"],
                           "[UtilPath]": ["/usr/bin/ls",
                                          "/usr/bin/sh",
                                          "/usr/bin/pwd",
                                          "/usr/bin/w",
                                          "/usr/bin/ls",
                                          "/usr/bin/ps",
                                          "/var/log/auth.log",
                                          "/var/log/syslog"]
                           }
RANDOM_NUM_PLACEHOLDERS = {"[Number]": [0, 999],
                           "[SmallNumber]": [1, 9],
                           "[MediumNumber]": [10, 99],
                           "[LargeNumber]": [100, 999],
                           "[PortNumber]": [1, 1024],
                           "[IPNumber]": [0, 255]
                           }
MAX_PH_EXEC_TRIES = 5

UNLEARNED_PLACEHOLDERS = LOCAL_RANDOM_PLACEHOLDERS + RANDOM_PLACEHOLDERS + list(RANDOM_NUM_PLACEHOLDERS.keys()) + list(RANDOM_NUM_PLACEHOLDERS.keys())
