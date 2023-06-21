# plots the rejected points on a frame

def plot_rejected(rejectedImgPoints, gray):
        
        rejectedImgPoints_x = []
        rejectedImgPoints_y = []
        
        for i in range(len(rejectedImgPoints)):
            temp_group = rejectedImgPoints[i]
            for j in range(len(temp_group)):
                temp_point = temp_group[j]
                for k in range(len(temp_point)):
                    temp_coord = temp_point[k]
                    for l in range(len(temp_coord)):
                        temp = temp_coord[l]
                        if l == 0:
                            rejectedImgPoints_x.append(temp)
                        else:
                            rejectedImgPoints_y.append(temp)

        plt.imshow(gray)
        plt.scatter(rejectedImgPoints_x, rejectedImgPoints_y)
        plt.show()
