import random as rd
import re
import math
import string
import numpy as np
import matplotlib.pyplot as plt
def cleandata(url):
    f = open(url, "r", encoding="utf8")
    tweets = list(f)
    list_of_tweets = []
    for i in range(len(tweets)):
        tweets[i] = tweets[i].strip('\n')
        tweets[i] = tweets[i][50:]
        tweets[i] = " ".join(filter(lambda x: x[0] != '@', tweets[i].split()))
        tweets[i] = re.sub(r"http\S+", "", tweets[i])
        tweets[i] = re.sub(r"www\S+", "", tweets[i])
        tweets[i] = tweets[i].strip()
        tweet_len = len(tweets[i])
        if tweet_len > 0:
            if tweets[i][len(tweets[i]) - 1] == ':':
                tweets[i] = tweets[i][:len(tweets[i]) - 1]
        tweets[i] = tweets[i].replace('#', '')
        tweets[i] = tweets[i].lower()
        tweets[i] = tweets[i].translate(str.maketrans('', '', string.punctuation))
        tweets[i] = " ".join(tweets[i].split())
        list_of_tweets.append(tweets[i].split(' '))
    f.close()
    return list_of_tweets
def kmeans(tweets, k=4, iterations=50):
    centroids = []
    sse=[]
    count = 0
    map = dict()
    while count < k:
        random_tweet = rd.randint(0, len(tweets) - 1)
        if random_tweet not in map:
            count += 1
            map[random_tweet] = True
            centroids.append(tweets[random_tweet])
    another_count = 0
    previous_centroids = []
    while (is_converged(previous_centroids, centroids)) == False and (another_count < iterations):
        print(" iteration " + str(another_count))
        clusters = assign_cluster(tweets, centroids)
        previous_centroids = centroids
        centroids = update_centroids(clusters)
        another_count = another_count + 1
        if (another_count == iterations):
            print(" iterations , Kmeans not convert ")
        else:
            print("convert")
        es = SSE(clusters)
        sse.append(es)
    return clusters, sse
def is_converged(previous_centroid, new_centroids):
    if len(previous_centroid) != len(new_centroids):
        return False
    for j in range(len(new_centroids)):
        if " ".join(new_centroids[j]) != " ".join(previous_centroid[j]):
            return False
    return True
def assign_cluster(tweets, centroids):
    clusters = dict()
    for t in range(len(tweets)):
        min_dis = math.inf
        cluster_idx = -1;
        for c in range(len(centroids)):
            dis = Jaccard(centroids[c], tweets[t])
            if centroids[c] == tweets[t]:
                cluster_idx = c
                min_dis = 0
                break
            if dis < min_dis:
                cluster_idx = c
                min_dis = dis
        if min_dis == 1:
            cluster_idx = rd.randint(0, len(centroids) - 1)
        clusters.setdefault(cluster_idx, []).append([tweets[t]])
        last_tweet_idx = len(clusters.setdefault(cluster_idx, [])) - 1
        clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(min_dis)
    return clusters
def update_centroids(clusters):
    centroids = []
    for c in range(len(clusters)):
        min_dis_sum = math.inf
        centroid_idx = -1
        min_dis_dp = []
        for t1 in range(len(clusters[c])):
            min_dis_dp.append([])
            dis_sum = 0
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    if t2 < t1:
                        dis = min_dis_dp[t2][t1]
                    else:
                        dis = Jaccard(clusters[c][t1][0], clusters[c][t2][0])
                    min_dis_dp[t1].append(dis)
                    dis_sum += dis
                else:
                    min_dis_dp[t1].append(0)
            if dis_sum < min_dis_sum:
                min_dis_sum = dis_sum
                centroid_idx = t1
        centroids.append(clusters[c][centroid_idx][0])
    return centroids
def Jaccard(tweet1, tweet2):
    intersection = set(tweet1).intersection(tweet2)
    union = set().union(tweet1, tweet2)
    return 1 - (len(intersection) / len(union))
def SSE(clusters):
    SSE = 0
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            SSE = SSE + (clusters[c][t][1] * clusters[c][t][1])
    return SSE
if __name__ == '__main__':
    data_url = 'bbchealth.txt'
    tweets = cleandata(data_url)
    experiments=5
    k=3
    for e in range(experiments):
        print("Kmeans for experiment  " + str((e + 1)) + " for k = " + str(k))
        clusters, sse = kmeans(tweets, k)
        for c in range(len(clusters)):
            print(str(c+1) + ": ", str(len(clusters[c])) + " tweets")
        print("SSE : " + str(sse))
        print('\n')
        x_ax = np.arange(1,len(sse)+1,1)
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('ggplot')
        ax.set_title('SSE')
        ax.set_ylabel('SSE')
        ax.set_xlabel('Iteration')
        ax.plot(x_ax, sse, marker='*', color='b', label='SSE (sum of squared error)')
        ax.legend(fontsize=9, loc='upper right')
        plt.show()
        k=k+1














