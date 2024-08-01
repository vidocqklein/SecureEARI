import math

def calculate_std_dev(num_list):
    # calculate the standard deviation of a list of numbers
    mean = sum(num_list) / len(num_list)
    mean = round(mean, 4)  # round to 4 decimal places
    # calculate variance using a list comprehension
    variance = sum((x - mean) ** 2 for x in num_list) / len(num_list)
    std_dev = math.sqrt(variance)
    std_dev = round(std_dev, 4)
    # calculate standard deviation
    return std_dev, mean

numbers = [73.63,  74.52,   63.66]
std, mean_value = calculate_std_dev(numbers)
print('std:', std)
print('mean', mean_value)