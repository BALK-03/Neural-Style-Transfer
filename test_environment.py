import sys

REQUIRED_PYTHON = (3, 10)

def check_environment():
    system_version = sys.version_info
    required_major, required_minor = REQUIRED_PYTHON

    if system_version.major != required_major or system_version.minor != required_minor:
        raise TypeError(
            "This project requires Python {}.{}. Found: Python {}.{}".format(
                required_major, required_minor, system_version.major, system_version.minor))
    else:
        print(">>> Development environment passes all tests!")

if __name__ == '__main__':
    check_environment()