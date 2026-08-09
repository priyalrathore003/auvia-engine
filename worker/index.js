addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const url = new URL(request.url)
  url.hostname = 'auvia-engine-754304552652.asia-south1.run.app'
  url.protocol = 'https:'

  return fetch(new Request(url.toString(), {
    method:   request.method,
    headers:  request.headers,
    body:     ['GET','HEAD'].includes(request.method) ? null : request.body,
    redirect: 'follow'
  }))
}
