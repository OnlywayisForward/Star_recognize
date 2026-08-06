# -*- coding: utf-8 -*-
"""重构版星图识别流水线。

模块划分：
    image.py           图像星点检测
    catalog.py         星表加载与解析
    geometry.py        坐标变换、角距离、Kabsch 旋转对齐
    camera.py          针孔相机模型（像素 -> 单位方向向量）
    database.py        星对哈希库（角距离 -> 星对）
    matching.py        三角形匹配、投票与全局验证
    visualization.py   结果标注与绘制
"""
