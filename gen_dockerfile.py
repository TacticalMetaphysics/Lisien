from http.client import HTTPSConnection


PYTHON_VERSIONS = ["3.12", "3.13", "3.14"]


with open("Dockerfile.head", "rt") as f:
	docker_head = f.read()
with open("Dockerfile.foot", "rt") as f:
	docker_foot = f.read()

url_template = "https://raw.githubusercontent.com/docker-library/python/refs/heads/master/{}/trixie/Dockerfile"
conn = HTTPSConnection("raw.githubusercontent.com")
with open("Dockerfile", "w") as f:
	f.write(docker_head)
	for version in PYTHON_VERSIONS:
		conn.request("GET", url_template.format(version))
		resp = conn.getresponse()
		docker_body = resp.read().decode()
		start = docker_body.index("ENV PYTHON_VERSION")
		end = docker_body.rindex("# make some useful symlinks that are expected to exist")
		f.write(docker_body[start:end])
	f.write(docker_foot)
