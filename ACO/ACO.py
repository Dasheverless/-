import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    #城市列表
    c = np.array([[1304 ,2312],[3639 ,1315],[4177 ,2244], [3712 ,1399],[3488 ,1535],[3326 ,1556],[3238 ,1229],[4196 ,1004],
        [4312 ,790],[4386 ,570],[3007 ,1970],[2562 ,1756],[2788 ,1491],[2381 ,1676],[1332 ,695],[3715 ,1678],[3918 ,2179],
        [4061 ,2370],[3780 ,2212],[3676 ,2578],[4029 ,2838],[4263 ,2931],[3429 ,1908],[3507 ,2376],[3394 ,2643],[3439 ,3201],
        [2935 ,3240],[3140 ,3550],[2545 ,2357],[2778 ,2826],[2370 ,2975]])
    n = c.shape[0]

    #参数设置
    m = 50
    alpha = 1
    beta = 5
    rho = 0.1
    G = 200 #最大迭代次数
    #变量设置
    Q = 100
    d = np.ones((n,n))*np.Inf #距离矩阵
    Tabu = np.zeros((m,n),int) #禁忌表
    Tau = np.ones((n,n)) #信息素
    l_best = np.zeros(G) #每次迭代最短路径值
    path_best = np.zeros((G,n),int)#每次迭代最短路径走法
    #计算距离
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d[i][j] = np.linalg.norm(c[i]-c[j])
    eta = 1/d
    #开始迭代
    for g in range(G):
        rand = np.array(np.random.randint(0,n,m))
        #随机将m只蚂蚁放在n个城市中
        for i in range(m):
            Tabu[i][0] = rand[i]
        #每只蚂蚁都需要完成一次城市的遍历才开始更新信息素
        for j in range(1,n):
            for i in range(m):
                P = np.zeros(n-j+1)
                visited = Tabu[i][:j] #已访问城市列表
                current_city = visited[-1]
                not_visited = np.setdiff1d(range(n),visited)#未访问城市列表
                for k in range(len(not_visited)):
                    #计算未访问城市的访问概率
                    P[k] = (Tau[current_city][not_visited[k]]**alpha)*(eta[current_city][not_visited[k]]**beta)
                P = P/np.sum(P)
                #计算累和是为了依据rand值能选出一个解
                P = np.cumsum(P)
                rand = np.random.uniform()
                next_city = np.where(P >= rand)[0][0]
                Tabu[i][j] = not_visited[next_city]
        #每只蚂蚁的路径值
        l = np.zeros(m)
        for i in range(m):
            #每只蚂蚁的路径走法
            Path = Tabu[i]
            for j in range(1,n):
                l[i] = l[i] + d[Path[j-1]][Path[j]]
            l[i] = l[i] + d[0][Path[n-1]]
            l_best[g] = min(l)
            path_best[g] = Tabu[np.where(l==l_best[g])[0][0]]
        delta_tau = np.zeros((n,n))
        #计算信息素增量
        for i in range(m):
            for j in range(1,n):
                delta_tau[Tabu[i][j-1]][Tabu[i][j]] +=  Q/l[i]
            delta_tau[Tabu[i][n-1]][Tabu[i][0]] +=  Q/l[i]
        #更新信息素
        Tau = (1-rho)*Tau + delta_tau
        #下一次迭代前需要清空禁忌表
        Tabu.fill(0)
    #显示结果
    print(min(l_best))
    min_index = np.where(l_best == min(l_best))[0][0]
    best = path_best[min_index]
    print(path_best[min_index])
    #绘制路径
    for i in range(1,len(best)):
        p1 = c[best[i-1]]
        plt.text(p1[0],p1[1],str(i))
        p2 = c[best[i]] 
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],color='r')
        plt.scatter(p1[0],p1[1],color='b')
    p1 = c[best[-1]]
    p2 = c[best[0]]
    plt.text(p1[0],p1[1],str(n))
    plt.plot([p1[0],p2[0]],[p1[1],p2[1]],color='r')
    plt.scatter(p1[0],p1[1],color='b')
    plt.show()

