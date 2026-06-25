print("\n" + "=" * 40)
print("演示 zip 配对")
print("=" * 40)

list_a = [1, 3, 2, 1, 5, 3]
list_b = [3, 2, 1, 5, 3]

print(f"list_a: {list_a}")
print(f"list_b: {list_b}")
print()

# zip 把两个列表按位置一对一配对
print("zip(list_a, list_b) 的结果:")
for pair in zip(list_a, list_b):
    print(f"  {pair}")

print(f"\n转成列表: {list(zip(list_a, list_b))}")
print()

# 关键规则：以最短的列表为准！
short = [1, 2, 3]
long = [10, 20, 30, 40, 50]
print(f"短列表: {short}")
print(f"长列表: {long}")
print(f"zip(短, 长): {list(zip(short, long))}")
print("→ 只配出 3 对，40 和 50 被丢弃了")
print()

# 三个列表也能 zip
list_c = [100, 200, 300, 400]
list_d = ["a", "b", "c", "d"]
list_e = [True, False, True, False]
print(f"三个列表配对: {list(zip(list_c, list_d, list_e))}")
