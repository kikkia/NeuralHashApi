# NeuralHashApi
[Public api site](https://hash.kikkia.dev)  
Basic http api to run neural hash on a given image and return the result


The point of this is to allow me to utilize a python implementation of neuralhash without needing to do it locally in other projects.
### This is a WIP

## Run it yourself
I package a docker image at `registry.gitlab.com/kikkia/neuralhashapi:latest`

You can use a docker compose file to spin up your own like so:
```yaml
version: '3.7'
services:
  api:
    image: registry.gitlab.com/kikkia/neuralhashapi:latest
    restart: always
    container_name: neuralhashApi
    environment: 
     - SERVER_PORT=80
    ports:
      - 80:80
```

### Configurable env variables
`SERVER_PORT` - Sets the port for the webserver. `Default: 80`  
`DD_ENABLE` - Enable Datadog tracing. (DD Agent on server required, may need to set `network_mode` to host in `docker`) `Default: False`
