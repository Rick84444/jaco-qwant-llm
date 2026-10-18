// Native C++ llama.cpp completion benchmark. No fake acceleration.
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

std::string request(int port, const std::string& body, int timeout) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) throw std::runtime_error("socket failed");
  timeval tv{timeout,0};
  setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof(tv));
  setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof(tv));
  sockaddr_in addr{};
  addr.sin_family=AF_INET;
  addr.sin_port=htons(static_cast<uint16_t>(port));
  inet_pton(AF_INET,"127.0.0.1",&addr.sin_addr);
  if (connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr)) != 0) {
    close(fd); throw std::runtime_error("cannot connect to llama-server");
  }
  const std::string msg="POST /completion HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: "
     + std::to_string(body.size())+"\r\nConnection: close\r\n\r\n"+body;
  size_t offset=0;
  while (offset<msg.size()) {
    ssize_t n=send(fd,msg.data()+offset,msg.size()-offset,0);
    if (n<=0) {close(fd);throw std::runtime_error("socket send failed");}
    offset+=static_cast<size_t>(n);
  }
  std::string result;
  char buffer[4096];
  while (true) {
    ssize_t n=recv(fd,buffer,sizeof(buffer),0);
    if (n==0) break;
    if (n<0) {close(fd);throw std::runtime_error("response timed out");}
    result.append(buffer,static_cast<size_t>(n));
    if(result.size()>1024*1024) {close(fd);throw std::runtime_error("response too large");}
  }
  close(fd);
  if(result.rfind("HTTP/1.",0)!=0 || result.find(" 200 ")==std::string::npos)
    throw std::runtime_error("upstream returned non-200 HTTP response");
  size_t split=result.find("\r\n\r\n");
  if(split==std::string::npos) throw std::runtime_error("invalid HTTP response");
  return result.substr(split+4);
}
double metric(const std::string& data,const std::string& key) {
  std::regex pattern("\""+key+"\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
  std::smatch found;
  if(!std::regex_search(data,found,pattern))
    throw std::runtime_error("missing metric: "+key);
  return std::stod(found[1].str());
}
int main(int argc,char**argv) {
  int port=11435,reps=1,predict=4,timeout=600;
  try {
    for(int i=1;i<argc;++i) {
      std::string arg=argv[i];
      if(arg=="--port"&&i+1<argc) port=std::stoi(argv[++i]);
      else if(arg=="--reps"&&i+1<argc) reps=std::stoi(argv[++i]);
      else if(arg=="--predict"&&i+1<argc) predict=std::stoi(argv[++i]);
      else if(arg=="--timeout"&&i+1<argc) timeout=std::stoi(argv[++i]);
      else throw std::runtime_error("usage: native_probe [--port 11435] [--reps 2] [--predict 4] [--timeout 600]");
    }
    if(port<1||port>65535||reps<1||reps>20||predict<1||predict>128||timeout<5||timeout>3600)
      throw std::runtime_error("invalid parameters");
    std::string body="{\"prompt\":\"2+2=\",\"n_predict\":"+std::to_string(predict)+
      ",\"stream\":false,\"cache_prompt\":false,\"temperature\":0,\"seed\":42}";
    std::cout<<"run,prompt_tokens,prompt_tps,generated_tokens,generated_tps\n";
    for(int i=0;i<reps;++i) {
      std::string r=request(port,body,timeout);
      std::cout<<i+1<<","<<metric(r,"prompt_n")<<","
        <<std::fixed<<std::setprecision(4)<<metric(r,"prompt_per_second")<<","
        <<metric(r,"predicted_n")<<","<<metric(r,"predicted_per_second")<<"\n";
    }
  } catch(const std::exception&e) {
    std::cerr<<"JACO native probe: "<<e.what()<<"\n";
    return 1;
  }
  return 0;
}
