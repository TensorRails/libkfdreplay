/* Minimal test helpers. */
#ifndef KFDR_TEST_H
#define KFDR_TEST_H

#include <stdio.h>
#include <stdlib.h>

static int test_failures;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            test_failures++;                                                     \
        }                                                                        \
    } while (0)

#define CHECK_EQ_U64(a, b)                                                                     \
    do {                                                                                       \
        unsigned long long _a = (unsigned long long)(a), _b = (unsigned long long)(b);         \
        if (_a != _b) {                                                                        \
            fprintf(stderr, "%s:%d: %s == %s failed: 0x%llx != 0x%llx\n", __FILE__, __LINE__, #a, #b, _a, _b); \
            test_failures++;                                                                   \
        }                                                                                      \
    } while (0)

#define TEST_DONE(name)                                                         \
    do {                                                                        \
        if (test_failures) {                                                    \
            fprintf(stderr, "%s: %d failure(s)\n", name, test_failures);        \
            return 1;                                                           \
        }                                                                       \
        printf("%s: ok\n", name);                                               \
        return 0;                                                               \
    } while (0)

#endif
