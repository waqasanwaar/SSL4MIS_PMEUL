# Semi-supervised Medical Image Segmentation via PerturbationAware Mutual Learning and Edge-Aware Uncertainty Loss for Accurate Anatomical Delineation

![Image](https://private-user-images.githubusercontent.com/4428728/624682172-7376eb86-eb56-47b9-97e5-7904d87851e1.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3ODQ2NjMzNDMsIm5iZiI6MTc4NDY2MzA0MywicGF0aCI6Ii80NDI4NzI4LzYyNDY4MjE3Mi03Mzc2ZWI4Ni1lYjU2LTQ3YjktOTdlNS03OTA0ZDg3ODUxZTEucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDcyMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjA3MjFUMTk0NDAzWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZDc3MmM4ZDExNzMzODg3YzI5NzdjZjdmMTBlNDRhNjllYWFiYTFjYjk2NmM5MWZjZDMzOWVhOWUxM2E4MTNmYiZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmcmVzcG9uc2UtY29udGVudC10eXBlPWltYWdlJTJGcG5nIn0.U05jHSlaQJSVictbfPi3-2bAQ9I6hCvW7Lr7ff5oSgU)

This is the official pytorch implementation of our method. The full paper is available at: [Paper](https://link.springer.com/article/10.1007/s12539-026-00862-3)

## Procedure

For training or validation on ACDC (2D) dataset, run the following commands:

```
python ./code/train_mcnet_2d_sdf.py  --model mcnet2d_v2 --labelnum 14 --gpu 0 
python ./code/train_mcnet_2d_sdf.py  --model mcnet2d_v2 --labelnum 7 --gpu 0  
python ./code/train_mcnet_2d_sdf.py  --model mcnet2d_v1 --labelnum 14 --gpu 0 
python ./code/train_mcnet_2d_sdf.py  --model mcnet2d_v1 --labelnum 7 --gpu 0  

python ./code/test_2d.py --exp MCNet_sdf --model mcnet2d_v2 --labelnum 7 --gpu 0 
python ./code/test_2d.py --exp MCNet_sdf --model mcnet2d_v2 --labelnum 14 --gpu 0 
python ./code/test_2d.py --exp MCNet_sdf --model mcnet2d_v1 --labelnum 7 --gpu 0 
python ./code/test_2d.py --exp MCNet_sdf --model mcnet2d_v1 --labelnum 14 --gpu 0

```
For training or validation on LA (3D) dataset, run the following commands:
```
python ./code/train_mcnet_3d_sdf.py --model mcnet3d_v2 --labelnum 16 --gpu 0 
python ./code/train_mcnet_3d_sdf.py --model mcnet3d_v2 --labelnum 8 --gpu 0 
python ./code/train_mcnet_3d_sdf.py --model mcnet3d_v1 --labelnum 16 --gpu 0 
python ./code/train_mcnet_3d_sdf.py --model mcnet3d_v1 --labelnum 8 --gpu 0 

python ./code/test_3d.py --model mcnet3d_v2 --exp MCNet_sdf --labelnum 16 --gpu 0 
python ./code/test_3d.py --model mcnet3d_v2 --exp MCNet_sdf --labelnum 8 --gpu 0 
python ./code/test_3d.py --model mcnet3d_v1 --exp MCNet_sdf --labelnum 16 --gpu 0 
python ./code/test_3d.py --model mcnet3d_v1 --exp MCNet_sdf --labelnum 8 --gpu 0 

```
## Reference

If you find our paper/work/code useful then please cite:
```http
Anwaar, W., Manh, V., Xue, W. et al. Semi-supervised Medical Image Segmentation via Perturbation-Aware Mutual Learning and Edge-Aware Uncertainty Loss for Accurate Anatomical Delineation. Interdiscip Sci Comput Life Sci (2026). https://doi.org/10.1007/s12539-026-00862-3
```
