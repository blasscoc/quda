#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

static int (*real_socket)(int, int, int) = NULL;
static int (*real_bind)(int, const struct sockaddr *, socklen_t) = NULL;
static int (*real_connect)(int, const struct sockaddr *, socklen_t) = NULL;
static int (*real_accept)(int, struct sockaddr *, socklen_t *) = NULL;
static int (*real_accept4)(int, struct sockaddr *, socklen_t *, int) = NULL;
static int (*real_getsockname)(int, struct sockaddr *, socklen_t *) = NULL;
static int (*real_getpeername)(int, struct sockaddr *, socklen_t *) = NULL;
static int (*real_setsockopt)(int, int, int, const void *, socklen_t) = NULL;
static int (*real_getsockopt)(int, int, int, void *, socklen_t *) = NULL;
static int (*real_close)(int) = NULL;

#define MAX_FDS 4096
struct fd_info {
    int is_inet_shim;
    int domain;      // AF_INET or AF_INET6
    uint16_t port;   // host byte order
};
static struct fd_info fd_table[MAX_FDS];
static pthread_mutex_t fd_mu = PTHREAD_MUTEX_INITIALIZER;

static void init_syms(void) {
    if (!real_socket) {
        real_socket = dlsym(RTLD_NEXT, "socket");
        real_bind = dlsym(RTLD_NEXT, "bind");
        real_connect = dlsym(RTLD_NEXT, "connect");
        real_accept = dlsym(RTLD_NEXT, "accept");
        real_accept4 = dlsym(RTLD_NEXT, "accept4");
        real_getsockname = dlsym(RTLD_NEXT, "getsockname");
        real_getpeername = dlsym(RTLD_NEXT, "getpeername");
        real_setsockopt = dlsym(RTLD_NEXT, "setsockopt");
        real_getsockopt = dlsym(RTLD_NEXT, "getsockopt");
        real_close = dlsym(RTLD_NEXT, "close");
    }
}

static void make_un_addr(struct sockaddr_un *un, uint16_t port) {
    memset(un, 0, sizeof(*un));
    un->sun_family = AF_UNIX;
    snprintf(un->sun_path, sizeof(un->sun_path), "/tmp/nsight_loopback_%u.sock", (unsigned)port);
}

static void fill_fake_inet(struct sockaddr *addr, socklen_t *addrlen, int domain, uint16_t port) {
    if (!addr || !addrlen) return;
    if (domain == AF_INET6) {
        struct sockaddr_in6 sin6;
        memset(&sin6, 0, sizeof(sin6));
        sin6.sin6_family = AF_INET6;
        sin6.sin6_port = htons(port);
        sin6.sin6_addr = in6addr_loopback;
        socklen_t copylen = (*addrlen < sizeof(sin6)) ? *addrlen : sizeof(sin6);
        memcpy(addr, &sin6, copylen);
        *addrlen = sizeof(sin6);
    } else {
        struct sockaddr_in sin;
        memset(&sin, 0, sizeof(sin));
        sin.sin_family = AF_INET;
        sin.sin_port = htons(port);
        sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        socklen_t copylen = (*addrlen < sizeof(sin)) ? *addrlen : sizeof(sin);
        memcpy(addr, &sin, copylen);
        *addrlen = sizeof(sin);
    }
}

int socket(int domain, int type, int protocol) {
    init_syms();
    if (domain == AF_INET || domain == AF_INET6) {
        int base_type = type & ~(SOCK_NONBLOCK | SOCK_CLOEXEC);
        int flags = type & (SOCK_NONBLOCK | SOCK_CLOEXEC);
        int fd = real_socket(AF_UNIX, base_type | flags, 0);
        if (fd >= 0 && fd < MAX_FDS) {
            pthread_mutex_lock(&fd_mu);
            fd_table[fd].is_inet_shim = 1;
            fd_table[fd].domain = domain;
            fd_table[fd].port = 0;
            pthread_mutex_unlock(&fd_mu);
        }
        return fd;
    }
    return real_socket(domain, type, protocol);
}

static uint16_t extract_port(const struct sockaddr *addr) {
    if (!addr) return 0;
    if (addr->sa_family == AF_INET) {
        const struct sockaddr_in *sin = (const struct sockaddr_in *)addr;
        return ntohs(sin->sin_port);
    } else if (addr->sa_family == AF_INET6) {
        const struct sockaddr_in6 *sin6 = (const struct sockaddr_in6 *)addr;
        return ntohs(sin6->sin6_port);
    }
    return 0;
}

int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        uint16_t port = extract_port(addr);
        struct sockaddr_un un;
        if (port == 0) {
            for (int attempt = 0; attempt < 1000; attempt++) {
                uint16_t cand = (uint16_t)(40000 + ((getpid() * 37 + attempt) % 20000));
                make_un_addr(&un, cand);
                if (access(un.sun_path, F_OK) != 0) {
                    if (real_bind(sockfd, (struct sockaddr *)&un, sizeof(un)) == 0) {
                        pthread_mutex_lock(&fd_mu);
                        fd_table[sockfd].port = cand;
                        pthread_mutex_unlock(&fd_mu);
                        return 0;
                    }
                }
            }
            errno = EADDRINUSE;
            return -1;
        } else {
            make_un_addr(&un, port);
            unlink(un.sun_path);
            int rc = real_bind(sockfd, (struct sockaddr *)&un, sizeof(un));
            if (rc == 0) {
                pthread_mutex_lock(&fd_mu);
                fd_table[sockfd].port = port;
                pthread_mutex_unlock(&fd_mu);
            }
            return rc;
        }
    }
    return real_bind(sockfd, addr, addrlen);
}

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        uint16_t port = extract_port(addr);
        struct sockaddr_un un;
        make_un_addr(&un, port);
        pthread_mutex_lock(&fd_mu);
        fd_table[sockfd].port = port;
        pthread_mutex_unlock(&fd_mu);
        return real_connect(sockfd, (struct sockaddr *)&un, sizeof(un));
    }
    return real_connect(sockfd, addr, addrlen);
}

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        struct sockaddr_un un;
        socklen_t unlen = sizeof(un);
        int cfd = real_accept(sockfd, (struct sockaddr *)&un, &unlen);
        if (cfd >= 0) {
            uint16_t p = fd_table[sockfd].port;
            int dom = fd_table[sockfd].domain;
            if (cfd < MAX_FDS) {
                pthread_mutex_lock(&fd_mu);
                fd_table[cfd].is_inet_shim = 1;
                fd_table[cfd].domain = dom;
                fd_table[cfd].port = p;
                pthread_mutex_unlock(&fd_mu);
            }
            fill_fake_inet(addr, addrlen, dom, p);
        }
        return cfd;
    }
    return real_accept(sockfd, addr, addrlen);
}

int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        struct sockaddr_un un;
        socklen_t unlen = sizeof(un);
        int cfd = real_accept4(sockfd, (struct sockaddr *)&un, &unlen, flags);
        if (cfd >= 0) {
            uint16_t p = fd_table[sockfd].port;
            int dom = fd_table[sockfd].domain;
            if (cfd < MAX_FDS) {
                pthread_mutex_lock(&fd_mu);
                fd_table[cfd].is_inet_shim = 1;
                fd_table[cfd].domain = dom;
                fd_table[cfd].port = p;
                pthread_mutex_unlock(&fd_mu);
            }
            fill_fake_inet(addr, addrlen, dom, p);
        }
        return cfd;
    }
    return real_accept4(sockfd, addr, addrlen, flags);
}

int getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        uint16_t p = fd_table[sockfd].port;
        if (p == 0) {
            struct sockaddr_un un;
            socklen_t unlen = sizeof(un);
            if (real_getsockname(sockfd, (struct sockaddr *)&un, &unlen) == 0) {
                unsigned parsed = 0;
                if (sscanf(un.sun_path, "/tmp/nsight_loopback_%u.sock", &parsed) == 1) {
                    p = (uint16_t)parsed;
                }
            }
            if (p == 0) p = 45678;
        }
        fill_fake_inet(addr, addrlen, fd_table[sockfd].domain, p);
        return 0;
    }
    return real_getsockname(sockfd, addr, addrlen);
}

int getpeername(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        uint16_t p = fd_table[sockfd].port ? fd_table[sockfd].port : 45678;
        fill_fake_inet(addr, addrlen, fd_table[sockfd].domain, p);
        return 0;
    }
    return real_getpeername(sockfd, addr, addrlen);
}

int setsockopt(int sockfd, int level, int optname, const void *optval, socklen_t optlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        if (level == IPPROTO_TCP || level == IPPROTO_IP || level == IPPROTO_IPV6) {
            return 0;
        }
        if (level == SOL_SOCKET && (optname == SO_REUSEADDR || optname == SO_REUSEPORT || optname == SO_KEEPALIVE)) {
            return 0;
        }
    }
    return real_setsockopt(sockfd, level, optname, optval, optlen);
}

int getsockopt(int sockfd, int level, int optname, void *optval, socklen_t *optlen) {
    init_syms();
    if (sockfd >= 0 && sockfd < MAX_FDS && fd_table[sockfd].is_inet_shim) {
        if (level == IPPROTO_TCP || level == IPPROTO_IP || level == IPPROTO_IPV6) {
            if (optval && optlen && *optlen >= sizeof(int)) {
                *(int *)optval = 0;
                *optlen = sizeof(int);
            }
            return 0;
        }
    }
    return real_getsockopt(sockfd, level, optname, optval, optlen);
}
