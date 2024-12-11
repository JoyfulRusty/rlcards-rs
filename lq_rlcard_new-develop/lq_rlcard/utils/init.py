import os


async def _do_start(server_class, server_name):
    await server_class.share_server(server_name).start_server()


async def start(server_class, server_name):
    assert server_class
    assert server_name
    if os.name == "nt":  # win
        server_name = server_name.replace("\\", "/")
    server_name = server_name.split("/")[-1]  # win下不支持转换???

    print(f"Start: python3 {server_name}.py")
    await _do_start(server_class, server_name)