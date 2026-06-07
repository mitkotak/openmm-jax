#ifndef OPENMM_JAX_BASE64_H_
#define OPENMM_JAX_BASE64_H_

#include "openmm/OpenMMException.h"
#include <cctype>
#include <string>
#include <vector>

namespace JaxPlugin {

static inline std::string encodeBase64(const std::string& input) {
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string output;
    output.reserve(((input.size()+2)/3)*4);
    int val = 0;
    int valb = -6;
    for (unsigned char c : input) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            output.push_back(chars[(val >> valb) & 0x3f]);
            valb -= 6;
        }
    }
    if (valb > -6)
        output.push_back(chars[((val << 8) >> (valb+8)) & 0x3f]);
    while (output.size()%4)
        output.push_back('=');
    return output;
}

static inline std::string decodeBase64(const std::string& input, const std::string& errorPrefix) {
    static const char chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<int> table(256, -1);
    for (int i = 0; i < 64; i++)
        table[static_cast<unsigned char>(chars[i])] = i;

    std::string output;
    output.reserve((input.size()*3)/4);
    int val = 0;
    int valb = -8;
    for (char ch : input) {
        unsigned char c = static_cast<unsigned char>(ch);
        if (c == '=')
            break;
        if (std::isspace(c))
            continue;
        if (table[c] < 0)
            throw OpenMM::OpenMMException(errorPrefix + ": invalid base64 data");
        val = (val << 6) + table[c];
        valb += 6;
        if (valb >= 0) {
            output.push_back(static_cast<char>((val >> valb) & 0xff));
            valb -= 8;
        }
    }
    return output;
}

} // namespace JaxPlugin

#endif
