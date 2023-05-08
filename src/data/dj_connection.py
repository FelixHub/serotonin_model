import datajoint as dj


def connect_noTLS() -> None:
    """
    Connect to database
    """
    dj.config["database.host"] = "0.0.0.0"
    dj.config["database.user"] = "root"
    dj.config["database.password"] = "simple"

    dj.conn(use_tls=False)
