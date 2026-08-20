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




def total(nums):
    total  = 0
    for num in nums :
        total += num
    print(total)

total([1,2,3,44,5,55])

def reverse (nums):
    print(nums[::-1])
reverse([1,23,4])




def count(nums,target_num):
    count = 0
    for num in nums:
        if num == target_num:
            count+=1
    print(f'target number {count} dafa aya he')

count([1,2,3,4,4,4,4,4,4,4,4,4],4)

def sort(nums):
    first = 0
    sorted = None
    for num in range(1,len(nums)):
        if nums[first] < nums[num] or nums[first] == nums[num]:
            first+=1
            sorted = True
        else :
            sorted = False
            break
    print(sorted)
sort([1,0,2,3])
def second_largest(nums):
    largest = 0
    second_largest = nums[0]
    for num in nums :
        if num < largest:
            largest = num
second_largest([9,8,7,6])





        

