

1.时间复杂度
算法的执行效率
算法的执行时间与算法的输入值之间的关系
def O1(num):
    i = num
    j = num*2
    return i + j

def OlogN(num):
    i = 1
    while (i<num):
        i = i*2
    return i

def ON(num):
    total = 0
    for i in range(num):
        total += i
    return total

def OMN(num1,num2):
    total = 0
    for i in range(num1):
        total += i
    for j in range(num2):
        total += j
    return total

def ONLogN(num1,num2):
    total = 0
    j = 0
    for i in range(num1):
        while(j<num2):
            total += i
            j = j * 2
    return total

def ON2(num):
    total = 0
    for i in range(num):
        for j in range(num):
            total += i + j
    return total
    
2.空间复杂度
算法存储空间与输入值之间的关系
# 空间复杂度O(1)
def test(num):
    total = 0
    for i in range(num):
        total += i
    return total
# 空间复杂度O(N)
def test(nums):
    array = []
    for num in nums:
        array.append(num)
    return array
    
3.【数据结构】数组
数组：在连续的内存空间中，存储一组相同类型的元素 [1,2,3]
#创建数组
a = []

# 添加元素
# 时间复杂度是O(1)或者是O(N)
a.append(1)
a.append(2)
a.append(3)
print(a)
# 时间复杂度是O(N)
a.insert(2,99)
print(a)

# 访问元素 用索引或下标访问元素
# 时间复杂度是O(1)
temp = a[2]
print(temp)

# 更新元素
# 时间复杂度是O(1)
a[2] = 88

# 删除元素 3种方法
# 时间复杂度是O(N)
a.remove(88) # 需要先遍历找到88这个元素，然后将其删掉
print(a)
a.pop(1)
print(a)
# 时间复杂度是O(1)
a.pop() # 删除最后一个元素
print(a)

# 获取数组长度
a = [1,2,3]
size = len(a)
print(size)

# 遍历数组
# 时间复杂度是O(N)
for i in a:
    print(i)
for index,element in enumerate(a):
    print("Index at：%s，is：%s" % (index,element))
for i in range(0,len(a)):
    print("when i is %s,element is %s" % (i,a[i]))

# 查找某个元素
# 时间复杂度是O(N)
index = a.index(2)
print(index)

# 数组排序
# 时间复杂度是O(NlogN)
a = [3,1,2]
# 从小到大
a.sort()
print(a)
# 从大到小
a.sort(reverse=True)
print(a)

# 力扣485
# 给定一个二进制数组 nums ， 计算其中最大连续 1 的个数。
def coun(nums):
    count = 0
    result = 0
    if nums is None or len(nums) == 0:
        return 0
    else:
        for i in nums:
            if i == 1:
                count += 1
            else:
                if result < count:
                    result = count
                count = 0
        if result < count:
            result = count
        return result

print(coun([1,1,0,1,1,1,0]))

# 力扣283
def func(nums):
    index = 0
    if nums is None or len(nums) == 0:
        return 0
    else:
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[index] = nums[i]
                index += 1
        for i in range(index,len(nums)):
            nums[i] = 0
        return nums

print(func([0,1,0,3,12]))

# 力扣27
# 双指针法
def func(nums,val):
    """
    将数组的某个元素移除，得到一个新的数组，返回新数组的长度
    :param nums: 一个数组
    :param val: 一个值
    :return: 一个新数组的长度
    """
    l = 0 # 左指针起点
    r = len(nums) - 1 # 右指针起点
    while(l < r):
        # 没有找到value，循环下去，直到找到value，然后跳出循环
        while(l < r and nums[l] != val):
            l += 1
        # 没有找到非value，循环下去，直到找到value，然后跳出循环
        while(l < r and nums[r] == val):
            r -= 1
        # 如果在左右两边分别找到value和非value，互换两者的位置
        nums[l], nums[r] = nums[r], nums[l]
    # 当l>=r，左右两指针相遇时，说明左指针之前的都是非value值，右指针之后的都是value值，移除元素的任务已经完成
    if nums[l] == val:
        return l
    else:
        return l + 1

4.【数据结构】链表
# 创建链表
from collections import deque
linkedlist = deque()

# 添加元素
# 时间复杂度：O(1)
linkedlist.append(1)
linkedlist.append(2)
linkedlist.append(3)
print(linkedlist)
# 时间复杂度：O(N) 从头走到尾
linkedlist.insert(2,99)
print(linkedlist)

# 访问元素
# 时间复杂度：O(N)
element = linkedlist[2]
print(element)

# 搜索元素
# 时间复杂度：O(N)
index = linkedlist.index(99)
print(index)

# 更新元素
# 时间复杂度O(N)
linkedlist[2] = 88
print(linkedlist)

# 删除元素
# 时间复杂度:O(N)
linkedlist.remove(88)
print(linkedlist)

# 链表长度
# 时间复杂度O(1)
length = len(linkedlist)
print(length)

# 力扣203
# 力扣答案
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy_head = ListNode(next=head) #添加一个虚拟节点
        cur = dummy_head
        while(cur.next!=None):
            if(cur.next.val == val):
                cur.next = cur.next.next #删除cur.next节点
            else:
                cur = cur.next
        return dummy_head.next
# 伪代码
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def removeElement(head,val):
    """
    移除链表中的元素，返回新的头节点
    :param head:头节点
    :param val:要移除的元素
    :return:新的头节点
    """
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy
    while head != None and head.next != None:
        if head.val == val:
            prev.next = head.next
            head = head.next
        else:
            prev = head
            head = head.next
    return dummy.next
    
# 力扣206
def reverseList(head):
    """
    反转链表，并返回反转后的链表
    :param head:头节点
    :return:反转后的链表
    """
    dummy = ListNode(0)
    dummy.next = head
    while head != None and head.next != None:
        dnext = dummy.next
        hnext = head.next
        dummy.next = hnext
        head.next = hnext.next
        hnext.next = dnext
    return dummy.next

5.【数据结构】队列
# 创建队列
from collections import deque
queue = deque()
# 添加元素
# 时间复杂度：O(1)
queue.append(1)
queue.append(2)
queue.append(3)
print(queue)
# 获取即将出队的元素
# 时间复杂度：O(1)
temp1 = queue[0]
print(temp1)
# 删除即将出队的元素
# 时间复杂度：O(1)
temp2 = queue.popleft()
print(temp2)
print(queue)
# 判断队列是否为空
len(queue) == 0
# 遍历队列
while len(queue) != 0:
    temp = queue.popleft()
    print(temp)
# 力扣933
from collections import deque

class RecentCounter:
    def __init__(self):
        self.q = deque()
    def ping(self, t:int) -> int:
        self.q.append(t)
        while len(self.q) > 0 and t - self.q[0] > 3000:
            self.q.popleft()
        return len(self.q)

6.【数据结构】栈Stack
class Test:
    def test(self):
        # 创建栈
        stack = []
        # 增加元素
        # 时间复杂度：O(1)
        stack.append(1)
        stack.append(2)
        stack.append(3)
        print(stack)
        # 获取栈顶元素
        # 时间复杂度：O(1)
        stack[-1]
        # 删除栈顶元素
        # 时间复杂度：O(1)
        temp = stack.pop()
        print(temp)
        # 获取栈的长度
        # 时间复杂度：O(1)
        len(stack)
        # 判断栈是否为空
        # 时间复杂度：O(1)
        len(stack) == 0
        # 遍历栈
        # 时间复杂度：O(N)
        while len(stack) > 0:
            temp = stack.pop()
            print(temp)
if __name__ == "__main__":
    test = Test()
    test.test()

# 力扣20
class Solution:
    def isValid(self, s:str) -> bool:
        """
        判断所给的字符串里的括号是否对应
        :param s:所给的字符串
        :return:字符串有效返回True，否则返回False
        """
        if len(s) == 0:
            return True
        stack = []
        for c in s:  # s = "([])]]"
            if c=='(' or c=='[' or c=='{': # 将左括号放在栈里面
                stack.append(c)
            else:
                if len(stack) == 0:
                    return False
                else:
                    temp = stack.pop()
                    if c==')':
                        if temp!='(':
                            return False
                    elif c==']':
                        if temp!='[':
                            return False
                    elif c=='}':
                        if temp!='{':
                            return False
        return  True if len(stack) == 0 else False
if __name__ == "__main__":
    so = Solution()
    s = "(()[]{})"
    print(so.isValid(s))

# 力扣496
class Solution:
    def nextGreaterElement(self, nums1, nums2):
        """
        找到对应数组的值的更大值，并将结果返回
        :param nums1: 一个数组
        :param nums2: 一个数组，数组nums1是其子集
        :return:
        """
        res = []
        stack = []
        for num in nums2:
            stack.append(num)
        for num in nums1:
            temp = []
            isFound = False
            nextMax = -1
            while (len(stack) !=0 and not isFound):
                top = stack.pop()
                if top > num:
                    nextMax = top
                elif top == num:
                    isFound = True
                temp.append(top)
            res.append(nextMax)
            while len(temp) != 0:
                stack.append(temp.pop())
        return res
if __name__ == "__main__":
    so = Solution()
    nums1 = [4,1,2]
    nums2 = [1,3,4,2]
    print(so.nextGreaterElement(nums1,nums2))

6.【数据结构】哈希表
class Test:
    def test(self):
        # 数组创建哈希表
        hashTable = ['']*4
        # 字典创建哈希表
        mapping = {}

        # 增加元素
        # 时间复杂度：O(1)
        hashTable[1] = 'hanmeimei'
        hashTable[2] = 'lihua'
        hashTable[3] = 'siyangyuan'
        mapping[1] = 'hanmeimei'
        mapping[2] = 'lihua'
        mapping[3] = 'siyangyuan'

        # 更新元素
        # 时间复杂度：O(1)
        hashTable[1] = 'bishi'
        mapping[1] = 'bishi'

        # 删除元素
        # 时间复杂度：O(1)
        hashTable[1] = ''
        mapping.pop(1)
        # del mapping[1]

        # 获取值
        # 时间复杂度：O(1)
        hashTable[3]
        mapping[3]

        # 键是否存在
        # 时间复杂度：O(1)
        # hashTable No
        3 in mapping

        # 哈希表的长度
        # 时间复杂度：O(1)
        # hashTable No
        len(mapping)

        # 是否为空
        # 时间复杂度：O(1)
        # hashTable No
        len(mapping) == 0

if __name__ == "__main__":
    test = Test()
    test.test()

# 力扣217
# set法
class Solution:
    def containsDuplicate(self, nums):
        if len(nums) == 0:
            return False
        hashset = set(nums)  # set会自动去重
        # 如果数组的长度和set后的长度相等，说明没有重复的元素
        return False if len(nums) == len(hashset) else True

# 力扣217
# 哈希表法
class Solution:
    def containsDuplicate(self, nums):
        if len(nums) == 0:
            return False
        mapping = {}
        for num in nums:
            if num not in mapping:
                mapping[num] = 1
            else:
                mapping[num] = mapping.get(num) + 1
        for v in mapping.values():
            if v > 1:
                return True
        return False 

# 力扣389
# 哈希表法
class Solution:
    def findTheDifference(self, s , t):
        if len(s) == 0:
            return t
        table = [0] * 26
        for i in len(t):
            if i < len(s):
                table[ord(s[i]) - ord('a')] -= 1
            table[ord(t[i]) - ord('a')] += 1
        for i in range(26):
            if table[i] != 0:
                return chr(i+97)
        return 'a'

# 力扣496
# 栈和哈希表
class Solution:
    def nextGreaterElement(self, nums1, nums2):
        """
        将数组里的每个元素及对应更大值，以键值对放在哈希表里面
        :param nums1: 数组
        :param nums2: 数组，nums1为其子集
        :return: 数组nums1里元素对应的更大值
        """
        res = []
        stack = []
        mapping = {}
        # 找到每个元素对应的更大值
        for num in nums2:
            # 后面的元素与栈顶的元素进行比较
            while len(stack) != 0 and num > stack[-1]:
                temp = stack.pop()
                mapping[temp] = num
            stack.append(num)
        while len(stack) != 0:
            mapping[stack.pop()] = -1
        for num in nums1:
            res.append(mapping[num])
        return res      
 
7.【数据结构】集合set
class Test():
    def test(self):
        # 创建集合
        s = set()

        # 添加元素
        # 时间复杂度O(1)
        s.add(10)
        s.add(3)
        s.add(5)
        s.add(2)
        s.add(2)
        print(s)

        # 删除元素
        # 时间复杂度O(1)
        s.remove(2)
        print(s)

        # 长度
        # 时间复杂度O(1)
        len(s)

if __name__ == '__main__':
    test = Test()
    test.test()
        
# 力扣705
class MyHashSet:
    # 数组
    # 空间复杂度 O(N)
    # 初始化 O(N)
    # 添加元素 O(1)
    # 删除元素 O(1)
    # 是否存在 O(1)

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashset = [0]*1000001

    def add(self, key):
        self.hashset[key] = 1

    def remove(self, key):
        self.hashset[key] = 0

    def contains(self, key):
        """
        Returns true if this set contains the specified element
        :param key:
        :return:
        """
        return self.hashset[key]

8.【数据结构】二叉树
import heapq

class Test:
    def test(self):
        # 创建最小堆
        minheap = []
        heapq.heapify(minheap) # 将数组堆化

        # 添加元素
        heapq.heappush(minheap, 10)
        heapq.heappush(minheap, 8)
        heapq.heappush(minheap, 9)
        heapq.heappush(minheap, 2)
        heapq.heappush(minheap, 1)
        heapq.heappush(minheap, 11)
        print(minheap)

        # 查看数据
        print(minheap[0])

        # 删除元素
        heapq.heappop(minheap)

        # 堆长
        len(minheap)

        # 遍历堆
        while len(minheap) != 0:
            print(heapq.heappop(minheap))

if __name__ == '__main__':
    so = Test()
    so.test()

# 力扣215
import heapq

class Solution:
    # Heap
    # N is the size of nums
    # 时间复杂度 O(NlogN)
    # 空间复杂度 O(N)
    def findKthLargest(self, nums, k):
        heap = []
        heapq.heapify(heap) # 将数组堆化
        for num in nums:
            heapq.heappush(heap, num*-1)
        
        while k > 1:
            heapq.heappop(heap)
            k = k - 1
            
        return heapq.heappop(heap)*-1
            

# 力扣692
import heapq
class Solution:
    # 堆和哈希表
    # N is the size of words
    # 时间复杂度 O(NlogK)
    # 空间复杂度 O(N)
    def topKFrequent(self, words, k):
        mapping = {}
        # 统计单词出现的次数
        for word in words:
            if word not in mapping:
                mapping[word] = 0
            mapping[word] = mapping[word] + 1

        print(mapping)
        heap = []
