RED = '\033[91m'
GREEN = '\033[92m'
GREEN2 = '\033[52m'
YELLOW  = '\033[33m'
BLUE    = '\033[34m'
MAGENTA = '\033[35m'
CYAN    = '\033[36m'
WHITE   = '\033[37m'
TEST = '\033[105m'
ENDC = '\033[0m'  # Resets the color to default

print(f"{GREEN}Ciao describe what you are looking for:{MAGENTA}\nType your response {TEST}here --> {ENDC}")
print(ENDC)

a = input(f"{MAGENTA}ciao cosa cerchi?\n{TEST}type here: ")
print(ENDC)
print('fine')

a = input(f"{MAGENTA}ciao cosa cerchi?\n{YELLOW}type here: ")
print(ENDC)
print('fine')