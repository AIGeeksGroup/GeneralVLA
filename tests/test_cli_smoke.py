from robot_memory_vla.app.main import build_parser


def test_parser_accepts_task_text() -> None:
    parser = build_parser()
    args = parser.parse_args(["--task", "抓起桌面上的瓶盖，放到右下角粉色盒子上"])
    assert args.task == "抓起桌面上的瓶盖，放到右下角粉色盒子上"


def test_parser_allows_preflight_without_task() -> None:
    parser = build_parser()
    args = parser.parse_args(["--preflight"])
    assert args.preflight is True
    assert args.task is None
