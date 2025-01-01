import matplotlib.pyplot as plt

def polt_one_model(model_name, res):
    
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"{model_name}_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(res["accuracy"], label=f"{model_name}_accuracy", marker="o")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f"{model_name}_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(res["test_loss"], label=f"{model_name}_loss", marker="o")
    plt.legend()
    
    plt.savefig(f"save_pic_path/{model_name}.png")
    plt.show()

def plot(RES):

    for model_name, res in RES.items():
        polt_one_model(model_name, res)

    # 绘制多个模型的准确率对比图
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    for model_name, res in RES.items():
        plt.plot(res["accuracy"], label=f"{model_name}_accuracy")

    plt.legend()
  
    # 绘制多个模型的损失对比图
    plt.subplot(1, 2, 2)    
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    for model_name, res in RES.items():
        plt.plot(res["test_loss"], label=f"{model_name}_test_loss")

    plt.legend()

    plt.savefig(f"save_pic_path/compare.png")
    plt.show()
