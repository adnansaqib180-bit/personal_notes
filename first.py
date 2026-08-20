# dsa practice 1929

def getConcatenation(nums : list):
    numbers = nums.copy()
    for i in nums:
        numbers.append(i)
    return numbers
print(getConcatenation([1,2,1]))

# dsa 29

def drop(num):
    numbers = set(num)
    nums = list(numbers)
    return len(nums)

print(drop([1,1,2]))