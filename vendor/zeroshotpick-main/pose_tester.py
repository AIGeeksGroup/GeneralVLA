#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易交互式位姿发送脚本，用于独立测试机械臂在相机坐标系下的 xyz / rpy / grip。

用法示例：
  python3 pose_tester.py --host 10.5.23.176 --port 8888

进入后按照提示输入：
  x y z rx ry rz grip
单位：
  - x, y, z: mm（相机坐标系下）
  - rx, ry, rz: deg
  - grip: 夹爪指令（与下位机现有语义一致）

输入空行直接回车可以沿用上一次的数值，只改你重新输入的那部分。
输入 q 或 quit 退出。
"""

import argparse
from typing import Optional, Tuple

from upper_client import UpperClient


def _parse_floats(
    s: str, prev: Optional[Tuple[float, float, float, float, float, float, float]]
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """
    将一行字符串解析为 7 个浮点数:
      x y z rx ry rz grip
    支持以下几种输入形式：
      1) 7 个数全部给出
      2) 给出部分数，其余沿用上一次的值（需要 prev 不为 None）
    例如：
      - 只想改 z: 输入 "    0   0  300"
      - 只想改 rz: 输入 "         0  0  0  -90"
    空行或仅空格返回 None，上层自行处理。
    """
    s = s.strip()
    if not s:
        return None

    parts = s.split()
    if len(parts) > 7:
        raise ValueError("最多只能输入 7 个数: x y z rx ry rz grip")

    vals = [float(p) for p in parts]

    if prev is None:
        if len(vals) != 7:
            raise ValueError("第一次输入需要提供完整的 7 个数: x y z rx ry rz grip")
        # 第一次完整输入
        return (
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
            float(vals[4]),
            float(vals[5]),
            float(vals[6]),
        )

    # 有上一次的值，可以只输入前缀
    x, y, z, rx, ry, rz, grip = prev
    merged = [x, y, z, rx, ry, rz, grip]
    for i, v in enumerate(vals):
        merged[i] = float(v)
    return tuple(merged)  # type: ignore[return-value]


def run_interactive(host: str, port: int) -> None:
    client = UpperClient(host=host, port=port)
    last_pose: Optional[Tuple[float, float, float, float, float, float, float]] = None

    print("\n=== 交互式位姿测试脚本 (pose_tester) ===")
    print(f"当前连接目标: {host}:{port}")
    print("单位说明: x,y,z=mm (相机坐标系), rx,ry,rz=deg, grip=夹爪指令\n")
    print("输入格式: x y z rx ry rz grip")
    print("例子:  28.12 -231.47 371.47  7.71 1.86 -98.87  0")
    print("提示:")
    print("  - 第一次输入必须给全 7 个数")
    print("  - 之后可以只输入前缀，缺省部分自动沿用上一次的值")
    print('  - 输入空行直接回车，仅重复发送上一次的位姿')
    print('  - 输入 "q" 或 "quit" 退出\n')

    try:
        while True:
            if last_pose is None:
                prompt = "输入 x y z rx ry rz grip (第一次必须给全 7 个数，q 退出): "
            else:
                lp = last_pose
                prompt = (
                    "输入 x y z rx ry rz grip (可只改前缀, 回车沿用上一次, q 退出)\n"
                    f"  上一次: x={lp[0]:.3f}, y={lp[1]:.3f}, z={lp[2]:.3f}, "
                    f"rx={lp[3]:.3f}, ry={lp[4]:.3f}, rz={lp[5]:.3f}, grip={lp[6]:.3f}\n"
                    "> "
                )

            try:
                line = input(prompt)
            except EOFError:
                print("\n收到 EOF，退出。")
                break

            if not line.strip():
                # 空行：重复上一次
                if last_pose is None:
                    print("当前还没有上一次位姿，请先输入一组完整的 7 个数。")
                    continue
            else:
                if line.strip().lower() in {"q", "quit"}:
                    print("退出。")
                    break

                try:
                    parsed = _parse_floats(line, last_pose)
                except ValueError as e:
                    print(f"[输入错误] {e}")
                    continue

                if parsed is None:
                    # 理论上不会到这里（前面已处理），防御性逻辑
                    print("解析失败，请重新输入。")
                    continue

                last_pose = parsed

            if last_pose is None:
                # 防御性判断
                print("内部错误：last_pose 为空，请重新输入。")
                continue

            x, y, z, rx, ry, rz, grip = last_pose
            print(
                f"\n[即将发送位姿] "
                f"x={x:.3f}, y={y:.3f}, z={z:.3f}, "
                f"rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}, grip={grip:.3f}"
            )

            try:
                client.send_pose(x, y, z, rx, ry, rz, grip)
            except Exception as e:
                print(f"[发送失败] {e}")
            else:
                print("[已发送]\n")
    finally:
        client.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="简易交互式位姿发送脚本，用于测试机械臂 xyz / rpy / grip"
    )
    p.add_argument("--host", default="10.5.23.176", help="下位机 IP")
    p.add_argument("--port", type=int, default=8888, help="下位机端口")
    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    run_interactive(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

