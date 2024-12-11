def solution(A):
	result = 0
	for number in A:
		# result ^= number
		result = result ^ number
	return result

print(solution([1, 2, 3, 4, 5, 6]))