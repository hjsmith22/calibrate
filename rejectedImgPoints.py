# plots the rejected points on a frame

rejectedImgPoints_x = []
rejectedImgPoints_y = []
framenum = 0
vidname = "ridCalibration_box01_20230608_15-51-56_leftmirror_1-470-270-920_flipped.avi"
good_frames = [5, 6, 7, 8, 9, 10, 12, 13, 16, 17, 18, 19, 20, 22, 24, 25, 28, 29, 30, 32, 80, 83, 84, 85, 86,
               87, 88, 89, 90, 100, 103, 104, 107, 111, 112, 113, 115, 121]

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
        framenum += 1
        framenum_str = "{}".format(framenum)

        plt.imshow(gray)
        plt.scatter(rejectedImgPoints_x, rejectedImgPoints_y)
        # plt.show()
        filename = vidname.rsplit(".", 1)[0] + '_' + framenum_str
                if framenum in good_frames:
                        image_path_out = r'C:\Users\haydenjs\Desktop\good_frames' + '\G' + filename + '.png'
                else:
                        image_path_out = r'C:\Users\haydenjs\Desktop\bad_frames' + '\G' + filename + '.png'
                
                data_path = r'C:\Users\haydenjs\Desktop\rejected_npys' + '\G' + filename + '.npy'
                image_path_in = r'C:\Users\haydenjs\Desktop\rejected_pngs' + '\G' + filename + '.png'
                
                np.save(data_path, gray)
                data = np.load(data_path)
                image = Image.fromarray(data)
                image.save(image_path_in)

                image = Image.open(image_path_in)
                image_array = np.array(image)
                fig, ax = plt.subplots()
                ax.imshow(image_array)
                ax.scatter(rejectedImgPoints_x, rejectedImgPoints_y, color='red', marker='o')
                
                plt.savefig(image_path_out)
                plt.close(fig)
