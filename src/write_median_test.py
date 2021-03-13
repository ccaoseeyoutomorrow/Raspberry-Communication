import numpy as np
import time


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.last = None


# 链表：节点管理类
class Link:
    def __init__(self):
        # 链表的表头
        self.head = Node(-1)

    def length(self):
        count = -1
        temp = self.head
        while temp:
            count += 1
            temp = temp.next
        return count

    def show(self):
        temp = self.head.next
        while temp:
            print(temp.data)
            temp = temp.next

    def median(self, length):
        len_half = int(length / 2)
        node_list = []
        temp = self.head.next
        while temp:
            node_list.append(temp.data)
            temp = temp.next
        node_list.sort()
        return node_list[len_half]

    def add(self, item):
        node = Node(item)
        if (self.head.next == None):
            node.last = self.head
            self.head.next = node
            return
        temp = self.head
        while (temp.next != None):
            temp = temp.next
        node.last = temp
        temp.next = node
        if (self.length() > data_length):
            self.head.next = self.head.next.next
            self.head.next.next.last = self.head

    def suit_data(self, length, yuzhi):
        len_half = int(length / 2)
        node_list = []
        temp = self.head.next
        while temp:
            node_list.append(temp.data)
            temp = temp.next
        node_list.sort()
        num = len_half
        for num in range(len_half, 0, -1):
            if num == 0:
                break
            if (node_list[num] - node_list[num - 1] <= yuzhi):
                continue
            break
        return node_list[num]


if __name__ == '__main__':
    # 创建链表
    data = np.load('../label_Data/4号树木x.npy')
    send_string = ''
    data_length = 10
    yuzhi = 0.002
    R_number = [(15, 13, 1), (21, 19, 2), (29, 23, 3), (33, 31, 4), (36, 32, 5), (26, 24, 6), (22, 18, 7), (16, 12, 8)]
    T_number = [(7, 0, 1), (11, 0, 2), (35, 0, 3), (37, 0, 4), (40, 0, 5), (38, 0, 6), (10, 0, 7), (8, 0, 8)]
    link_A = [[] for i in range(len(R_number))]
    for i in range(len(R_number)):
        for j in range(len(T_number)):
            link_A[i].append(Link())
    count = 0
    for times in range(int(data.shape[0] / data_length)):
        if data[times * data_length][29] != data[times * data_length + data_length - 1][29]:
            continue
        count += 1
    datatemp = [[] for i in range(count)]
    count = 0
    for times in range(int(data.shape[0] / data_length)):
        if data[times * data_length][29] != data[times * data_length + data_length - 1][29]:
            continue
        for dl in range(data_length):
            for r in range(len(R_number) - 1):
                for i in range(r + 1, len(T_number)):
                    sum = 0
                    for temp in range(r - 0):
                        sum += 7 - temp
                    time_difference = data[times * data_length + dl][i - r + sum - 1]
                    link_A[r][i].add(time_difference)
        for i in range(len(R_number) - 1):
            for j in range(i + 1, len(T_number)):
                datatemp[count].append(str(link_A[i][j].median(data_length)))
        for r in range(len(R_number) - 1):
            for i in range(r + 1, len(T_number)):
                link_A[r][i] = Link()
        for i in range(28, 44):
            datatemp[count].append(data[times * data_length][i])
        count += 1
    datatemp = np.array(datatemp)
    np.save('../label_Data/4号树木x_median.npy', datatemp)
