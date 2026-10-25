#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static bool file_exists(const std::string &p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

int main(int argc, char **argv) {
  std::string obj_cache;
  std::string gauge_update_o;
  std::string cpp_dir;
  std::string dslash_dir;
  std::string out_so;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--obj-cache" && i + 1 < argc) {
      obj_cache = argv[++i];
    } else if (a == "--gauge-update-o" && i + 1 < argc) {
      gauge_update_o = argv[++i];
    } else if (a == "--cpp-dir" && i + 1 < argc) {
      cpp_dir = argv[++i];
    } else if (a == "--dslash-dir" && i + 1 < argc) {
      dslash_dir = argv[++i];
    } else if (a == "--out-so" && i + 1 < argc) {
      out_so = argv[++i];
    }
  }

  if (obj_cache.empty() || gauge_update_o.empty() || cpp_dir.empty() || dslash_dir.empty() || out_so.empty()) {
    fprintf(stderr,
            "Usage: %s --cpp-dir <dir> --dslash-dir <dir> --out-so <path/to/libquda.so>\n",
            argv[0]);
    return 1;
  }

  std::string link_txt = obj_cache + "/lib/CMakeFiles/quda.dir/link.txt";
  std::ifstream fin(link_txt);
  if (!fin) {
    fprintf(stderr, "Cannot open %s\n", link_txt.c_str());
    return 2;
  }

  std::unordered_map<std::string, std::string> replacements;
  replacements["CMakeFiles/quda.dir/gauge_update_quda.cu.o"] = gauge_update_o;

  const char *cpp_units[] = {"hmc.cpp", "hmc_integrator.cpp",
                             "interface_quda.cpp", "inv_cg_quda.cpp",
                             "solve.cpp"};
  for (const char *u : cpp_units) {
    std::string candidate = cpp_dir + "/" + u + ".o";
    replacements[std::string("CMakeFiles/quda_cpp.dir/") + u + ".o"] =
        candidate;
    replacements[std::string("CMakeFiles/quda.dir/") + u + ".o"] = candidate;
  }

  const char *dslash_units[] = {
      "dslash_wilson_half_ncolor3_ddno_recon9or8.cu",
      "dslash_wilson_half_ncolor3_ddno_recon13or12.cu",
      "dslash_wilson_single_ncolor3_ddno_recon9or8.cu",
      "dslash_wilson_single_ncolor3_ddno_recon13or12.cu"};
  for (const char *u : dslash_units) {
    std::string candidate = dslash_dir + "/" + u + ".o";
    if (file_exists(candidate)) {
      replacements[std::string("CMakeFiles/quda.dir/") + u + ".o"] = candidate;
    }
  }

  std::vector<std::string> tokens;
  std::string tok;
  while (fin >> tok) {
    tokens.push_back(tok);
  }

  std::vector<std::string> args;
  for (size_t i = 0; i < tokens.size(); i++) {
    if (tokens[i] == "-o" && i + 1 < tokens.size()) {
      args.push_back("-o");
      args.push_back(out_so);
      i++;
      continue;
    }
    auto it = replacements.find(tokens[i]);
    if (it != replacements.end()) {
      args.push_back(it->second);
    } else {
      args.push_back(tokens[i]);
    }
  }

  std::string work_lib_dir = obj_cache + "/lib";
  if (chdir(work_lib_dir.c_str()) != 0) {
    perror("chdir to obj_cache/lib failed");
    return 3;
  }

  std::vector<char *> argv_exec;
  argv_exec.reserve(args.size() + 1);
  for (auto &s : args) {
    argv_exec.push_back(const_cast<char *>(s.c_str()));
  }
  argv_exec.push_back(nullptr);

  pid_t pid = fork();
  if (pid < 0) {
    perror("fork failed");
    return 4;
  }
  if (pid == 0) {
    execvp(argv_exec[0], argv_exec.data());
    perror("execvp linker failed");
    _exit(127);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    perror("waitpid failed");
    return 5;
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    fprintf(stderr, "Linker exited with non-zero status\n");
    return 6;
  }

  printf("[relink_quda] Successfully linked %s\n", out_so.c_str());
  return 0;
}
