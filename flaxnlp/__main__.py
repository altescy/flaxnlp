def run() -> None:
    import logging
    import os

    from flaxnlp.commands import main

    if os.environ.get("FLAXNLP_DEBUG"):
        LEVEL = logging.DEBUG
    else:
        level_name = os.environ.get("FLAXNLP_LOG_LEVEL", "INFO")
        LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=LEVEL)

    main(prog="flaxnlp")


if __name__ == "__main__":
    run()
