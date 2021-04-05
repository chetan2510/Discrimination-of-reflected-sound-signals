# Python3 program for the above approach
fib = [0] * 101
dp1 = [0] * 101
dp2 = [0] * 101
v = [0] * 101

# Function to generate the
# fibonacci number
def fibonacci():

	# First two number of
	# fibonacci sqequence
	fib[1] = 1
	fib[2] = 2

	for i in range(3, 87 + 1):
		fib[i] = fib[i - 1] + fib[i - 2]

# Function to find maximum ways to
# represent num as the sum of
# fibonacci number
def find(num):

	cnt = 0

	# Generate the Canonical form
	# of given number
	for i in range(87, 0, -1):
		if(num >= fib[i]):
			v[cnt] = i
			cnt += 1
			num -= fib[i]

	# Reverse the number
	v[::-1]

	# Base condition of dp1 and dp2
	dp1[0] = 1
	dp2[0] = (v[0] - 1) // 2

	# Iterate from 1 to cnt
	for i in range(1, cnt):

		# Calculate dp1[]
		dp1[i] = dp1[i - 1] + dp2[i - 1]

		# Calculate dp2[]
		dp2[i] = (((v[i] - v[i - 1]) // 2) *
				dp2[i - 1] +
				((v[i] - v[i - 1] - 1) // 2) *
				dp1[i - 1])

	# Return final ans
	return dp1[cnt - 1] + dp2[cnt - 1]

# Driver Code

# Function call to generate the
# fibonacci numbers
fibonacci()

# Given number
num = 30

# Function call
print(find(num))

# This code is contributed by Shivam Singh
