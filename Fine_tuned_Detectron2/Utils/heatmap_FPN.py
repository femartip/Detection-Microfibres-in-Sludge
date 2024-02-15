import matplotlib.pyplot as plt
import cv2

for j in range(930,1660):
    imgs = []
    imgs.append(cv2.imread("./Fine_tuned_Detectron2/data/Dataset/images/{}.jpg".format(j)))
    for i in range(5):  # five levels
        heatmap = cv2.imread('./Fine_tuned_Detectron2/data/objectness/' + str(j) + '_' + str(i) + '.png')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        imgs.append(cv2.resize(imgs[0], (320, 184)) // 2 + heatmap // 2)  # blending
    fig = plt.figure(figsize=(16, 7))
    for i, img in enumerate(imgs):
        fig.add_subplot(2, 3, i + 1)
        if i > 0:
            plt.imshow(img[0:-1, :, ::-1])  # ::-1 removes the edge
            plt.title("objectness on P" + str(i + 1))
        else:
            plt.imshow(img[:, :, ::-1])
            plt.title("input image")
    plt.savefig("./Fine_tuned_Detectron2/data/heatmap/{}.png".format(j))
    plt.close()