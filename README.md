# Keep Pigeons Away

Detect pigeons by using YOLOv3-tiny model that trained with darknet and plenty of pigeon pictures, and use videostream, 2 servomotors and a laser module to keep them away!

## Requirements

- A powerful linux server with multiple NVIDIA GPUs for model training
- A development board with NVIDIA GPU, GPIO and I2C support
- A camera with videostream support or webcam
- A laser module that can be controlled by GPIO
- A Adafruit 16-channel 12-bit PWM/servo driver, aka PCA9685
- 2 Tower Pro MG995 servomotors

## Preparations

1. Install CUDA 10.0, cuDNN 7.6.0, OpenCV 4.1.0 and CMake 3.14.4.
2. Clone and compile `darknet`, and copy all files and directories into `./darknet`.
3. Download [darknet53.conv.74](https://pjreddie.com/media/files/darknet53.conv.74) into `./darknet`.
4. Copy all pictures and their label files into `./darknet/pigeons_data/pigeons`.
5. List all pictures' path in `./darknet/pigeons_data/pigeons.names`.

## Train models

1. Execute `sh train_yolov3-tiny-pigeons.sh` to train the model above 1000 iterations first.
2. Execute `sh train_yolov3-tiny-pigeons_gpus.sh` to train the model with multiple GPUs.

## Configurations

In `./main.py`:

- Set `IS_TEST_NEEDED` to `True` when testing laser and sweep areas is needed.
- Set `IS_DECIDE_ONLY` to `True` for executing with video file or running in device which does not have GPIO and I2C.
- Set `IS_DETECT_ONLY` to `True` for executing YOLO detection only.

## Execution

- Execute `sudo python3 ./main.py <arguments>`, arguments are not required.
- Arguments include `-vi`, `-vo`, `-sw` and `-sh`.
- Use `-h` or `--help` argument to show the usage of above arguments.
