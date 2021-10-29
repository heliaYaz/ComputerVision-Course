import cv2


def save_frames(file_name):
    # getting frames
    cap = cv2.VideoCapture(file_name)

    cnt = 1

    while True:
        ret, frame = cap.read()
        save(cnt, frame)

        if cnt == 900:
            break
        cnt += 1

    cap.release()
    cv2.destroyAllWindows()
    return


def save(num, f):
    cv2.imwrite('frames/num'+str(num)+".jpg", f)
    return


save_frames('video.mp4')
