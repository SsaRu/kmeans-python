import numpy as np
import random
import math
import matplotlib.pyplot as plt
import copy


class Data:
    def __init__(self, sample=[[],[]]):
        self.x = sample[0]
        self.y = sample[1]

    def append(self, data):
        self.x.append(data.x)
        self.y.append(data.y)

def sample(N=1000, R=5):
    mu, sigma = [random.uniform(0, R), random.random()]
    x = [random.gauss(mu, sigma) for i in range(N)]
    y = [random.gauss(mu, sigma) for i in range(N)]

    return x, y

class kmeans:

    def __init__(self,K, data):
        #1. 센트로이드 쌍(x,y)를 임의로 할당하여 object K수 만큼 생성한다.
        #2. 해당 클러스터의 클래스를 object K 수만큼 생성한다.
        self.pre_centroid_x = [(random.random() ** 4) for i in range(K)]
        self.pre_centroid_y = [(random.random() ** 4) for i in range(K)]
        self.centroid_x = [(random.random()**4) for i in range(K)]
        self.centroid_y = [(random.random()**4) for i in range(K)]
        self.cluster_x = [ [] for i in range(K)]
        self.cluster_y = [[] for i in range(K)]
        self.d = [ [] for i in range(K)]
        self.data = data

    def init_plot(self):
        plt.scatter(self.data.x, self.data.y, marker='v', color='r')
        plt.scatter(self.centroid_x[0], self.centroid_y[0], marker='*', color='y', s=160)
        plt.scatter(self.centroid_x[1], self.centroid_y[1], marker='*', color='k', s=160)
        plt.scatter(self.centroid_x[2], self.centroid_y[2], marker='*', color='m', s=160)
        plt.show()

    def plot(self):

        plt.scatter(self.cluster_x[0], self.cluster_y[0], marker='v', color='r')
        plt.scatter(self.cluster_x[1], self.cluster_y[1], marker='v', color='g')
        plt.scatter(self.cluster_x[2], self.cluster_y[2], marker='v', color='b')
        plt.scatter(self.centroid_x[0], self.centroid_y[0], marker='*', color='y', s=160)
        plt.scatter(self.centroid_x[1], self.centroid_y[1], marker='*', color='k', s=160)
        plt.scatter(self.centroid_x[2], self.centroid_y[2], marker='*', color='m', s=160)
        plt.show()

    def re_init(self):
        for i in range(len(self.cluster_x)):
            self.cluster_x[i].clear()
            self.cluster_y[i].clear()

        self.assign_cluster()

    def check_finished(self):
        flags = 0
        for i in range(len(self.centroid_x)):
            flags += self.get_euclidean_distance(self.pre_centroid_x[i], self.pre_centroid_y[i],
                                                 self.centroid_x[i], self.centroid_y[i])

        self.plot()

        if flags < 3:
            self.plot()
            return True
        else:
            self.re_init()

    def move_centroid(self):
        print("-----MOVE CENTROID-----")
        for i in range(len(self.cluster_x)):
            sum_x = 0
            sum_y = 0
            for j in range(len(self.cluster_x[i])):
                sum_x += self.cluster_x[i][j]
                sum_y += self.cluster_y[i][j]

            print("\nsum x : {}".format(sum_x))
            print("length x : {}".format(len(self.cluster_x[i])))
            print("sum y : {}".format(sum_y))
            print("length y : {}".format(len(self.cluster_y[i])))

            if (len(self.cluster_x[i])==0 or len(self.cluster_y[i])==0):
                self.pre_centroid_x[i] = self.centroid_x[i]
                self.pre_centroid_y[i] = self.centroid_y[i]
                self.centroid_x[i] = self.centroid_x[i]
                self.centroid_y[i] = self.centroid_y[i]

                print("new centorid_x[{}] : {}".format(i, self.centroid_x[i]))
                print("new centorid_y[{}] : {}\n".format(i, self.centroid_y[i]))
            else:
                self.pre_centroid_x[i] = self.centroid_x[i]
                self.pre_centroid_y[i] = self.centroid_y[i]
                self.centroid_x[i] = sum_x / len(self.cluster_x[i])
                self.centroid_y[i] = sum_y / len(self.cluster_y[i])

                print("new centorid_x[{}] : {}".format(i, self.centroid_x[i]))
                print("new centorid_y[{}] : {}\n".format(i, self.centroid_y[i]))
        print("-----FINISHED MOVE CENTROID-----")
        self.check_finished()

    def assign_cluster(self):
        # centroid들과 임의의 data간의 유클리디언 거리가 제일 작다면,
        # 해당 centroid에 데이터를 할당한다.
        print("-----ASSIGN CLUSTER-----")
        if len(self.data.x) == 0 or len(self.data.y) == 0:
            print("data empty!!!!!")
        print("data x : {}".format(self.data.x))
        print("data y : {}".format(self.data.y))
        print("length of data x : {}".format(len(self.data.x)))
        print("length of data y : {}".format(len(self.data.y)))
        print("length of data d : {}".format(len(self.d)))
        print("length of cluster x : {}".format(len(self.cluster_x)))
        print("length of cluster y : {}".format(len(self.cluster_y)))
        print()
        for i in range(0, len(self.data.x)):
            print("---------calculate euclidean distance---------")
            for j in range(0, len(self.d)):
                print("data (x, y) : ({},{})".format(self.data.x[i], self.data.y[i]))
                print("centroid (x, y) : ({},{})".format(self.centroid_x[j], self.centroid_y[j]))

                self.d[j] = self.get_euclidean_distance(self.data.x[i], self.data.y[i],
                                                        self.centroid_x[j], self.centroid_y[j])
                print("Euclidean Distance : {}".format(self.d[j]))

            print("---------Finished calculate euclidean distance---------")
            print("---------compare with euclidean distance---------")
            min_distance = min(self.d[0], min(self.d[1], self.d[2]))
            index = self.d.index(min_distance)
            print("\nmin_distance : {}".format(min_distance))
            print("min_distance index : {}\n".format(index))
            print("---------assign cluster---------")
            print("selected index : Cluster[{}]".format(index))
            print("present index : data[{}]".format(i))
            self.cluster_x[index].append(data.x[i])
            self.cluster_y[index].append(data.y[i])
        print("-----FINISHED ASSIGN CLUSTER-----")
        self.move_centroid()

    def get_euclidean_distance(self, d_x, d_y, c_x, c_y):
        # 입력된 data와 centroid간 유클리디언 디스턴스를 구한다.
        return math.sqrt(math.pow((c_x-d_x),2) + math.pow((c_y-d_y),2))

a = Data(sample(N=1000, R=5))
b = Data(sample(N=1000, R=5))
c = Data(sample(N=1000, R=5))

data = Data()

data.x.extend(a.x)
data.y.extend(a.y)
data.x.extend(b.x)
data.y.extend(b.y)
data.x.extend(c.x)
data.y.extend(c.y)

plt.scatter(a.x,a.y, marker='.')
plt.scatter(b.x,b.y, marker='.', color='r')
plt.scatter(c.x,c.y, marker='.', color='g')
plt.show()

k_means = kmeans(3, data)
k_means.init_plot()
k_means.assign_cluster()



