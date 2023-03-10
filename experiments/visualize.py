from matplotlib import pyplot as plt
import numpy as np

x = [0.0035, 0.19, 1.43, 2.00, 4.11, 6.32, 8.67, 10.55]
x.reverse()
y1 = np.array([0.529, 0.556, 0.520, 0.534, 0.538, 0.620, 0.725, 0.656])
y2 = np.array([921, 896, 880, 841, 791, 911, 822, 757])/1000
y3 = np.array([966, 939, 935, 927, 918, 953, 939, 917])/1000
methods = ["HK1", "PPR0.80", "PPR0.85", "PPR0.90", "PPR0.95", "HK2", "HK3", "HK5"]

plt.scatter(x, y1, label="Highschool")
plt.xlabel("Radius")
plt.ylabel("Utility loss")
plt.ylim(bottom=[0, max(y1)+0.0003])
#splt.xscale('log')
for i, txt in enumerate(methods):
    plt.annotate(txt, (x[i]-0.25, y1[i]+0.00008))
plt.legend()
plt.show()

#plt.scatter(x, y2)
#plt.scatter(x, y3)
plt.xlabel("Radius")
plt.ylabel("AUC")
plt.ylim([0.5,1])
for i, txt in enumerate(methods):
    plt.plot([x[i], x[i]], [y2[i], y3[i]], 'o', linestyle="--")

#plt.plot([-1, max(x)+1], [0.93, 0.93], 'gray', linestyle="--")
#splt.xscale('log')
for i, txt in enumerate(methods):
    plt.annotate(txt, (x[i]-0.25, y3[i]+0.005))
    plt.annotate("▼", (x[i]-0.11, (y2[i]+y3[i])/2))
plt.show()