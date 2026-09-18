"""Bounded loopback transport probe; no model runtime or system settings change."""
import json,socket,threading,time
rows=[]
for timing in ['before_connect','after_connect','both']:
 listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
 stop=threading.Event();ready=threading.Event();counts=[]
 def serve():
  conn,_=listener.accept();conn.settimeout(0.2);ready.set();sent=0
  try:
   while not stop.is_set():
    try: sent+=conn.send(b'x'*4096)
    except socket.timeout:pass
  except (ConnectionError,OSError):pass
  finally:counts.append(sent);conn.close()
 thread=threading.Thread(target=serve);thread.start()
 client=socket.socket();client.settimeout(2)
 if timing in ['before_connect','both']:client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024)
 before=client.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
 client.connect(listener.getsockname());connected=client.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
 if timing in ['after_connect','both']:client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024)
 after=client.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
 assert ready.wait(2)
 client.recv(1);time.sleep(1)
 held=client.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
 stop.set();client.close();thread.join(3);listener.close();assert not thread.is_alive()
 rows.append(dict(timing=timing,before=before,connected=connected,after=after,held=held,sent=counts[0]))
print(json.dumps(rows,indent=2))
