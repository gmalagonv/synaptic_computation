# test.py

def testing(a, b, c):
    print('started')
    d = a + b + c
    print(d, 'working')

    return d

# If you want to call the testing function with specific arguments when the script is executed directly:
if __name__ == "__main__":
    import sys
    # The first argument (sys.argv[0]) is the script name
    # The subsequent arguments (sys.argv[1:]) are the command-line arguments
    args = sys.argv[1:]  # Ignore the script name

    # Convert command-line arguments to integers (assuming they are numeric)
    args = [int(arg) for arg in args]

    # Call the testing function with the provided command-line arguments
    result = testing(*args)
    print('Result:', result)
