from ultralytics import YOLO

if __name__ == '__main__':
    # モデルの読み込み
    model = YOLO('yolo11n.pt')

    # 画像のパスを指定して推論を行う
    # save=True：検知結果を別のファイルに保存
    # conf=0.5：信頼度が0.5以上の検知結果を保存
    model.predict('https://ultralytics.com/images/bus.jpg', save=True, conf=0.5)
