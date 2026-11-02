import requests

base = "http://172.24.109.143:8081"

paths = [
    "",
    "/video",
    "/mjpeg",
    "/live",
    "/stream",
    "/camera",
    "/snapshot",
    "/videofeed"
]

for p in paths:

    url = base + p

    try:

        r = requests.get(
            url,
            timeout=5,
            allow_redirects=True
        )

        print(
            url,
            r.status_code,
            r.headers.get("content-type")
        )

    except Exception as e:

        print(
            url,
            "ERROR",
            e
        )