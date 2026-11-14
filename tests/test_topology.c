/*
 * Unit tests: KFD sysfs topology parsing and topology-path rewriting, on a
 * fake sysfs tree. No GPU needed.
 */
#include "kfdr.h"
#include "test.h"

#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static char root[256], nodes[300], dri[300];

static void write_file(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(text, f);
    fclose(f);
}

/* One topology node; minor < 0 means a CPU node (gpu_id 0, no render minor). */
static void add_node(int n, unsigned gpu_id, int minor) {
    char d[512], p[600], props[512];
    snprintf(d, sizeof d, "%s/%d", nodes, n);
    mkdir(d, 0755);
    snprintf(p, sizeof p, "%s/gpu_id", d);
    char v[32];
    snprintf(v, sizeof v, "%u\n", gpu_id);
    write_file(p, v);
    snprintf(p, sizeof p, "%s/properties", d);
    if (minor >= 0)
        snprintf(props, sizeof props,
                 "cpu_cores_count 0\nsimd_count 1024\nlocation_id 1234\ndrm_render_minor %d\nhive_id 0\nnum_xcc 8\n", minor);
    else
        snprintf(props, sizeof props, "cpu_cores_count 64\nsimd_count 0\ndrm_render_minor 0\n");
    write_file(p, props);
}

static void add_render(int minor) {
    char p[400];
    snprintf(p, sizeof p, "%s/renderD%d", dri, minor);
    write_file(p, "");
}

static void rm_render(int minor) {
    char p[400];
    snprintf(p, sizeof p, "%s/renderD%d", dri, minor);
    unlink(p);
}

static void test_sysfs(void) {
    add_node(0, 0, -1); /* CPU */
    add_node(1, 111, 128);
    add_node(2, 222, 136);
    add_node(3, 333, 144);
    add_render(136); /* the container sees only node 2's render node */

    CHECK(kfdr_topo_node_of(222) == 2);
    CHECK(kfdr_topo_node_of(333) == 3);
    CHECK(kfdr_topo_node_of(999) == -1);

    CHECK_EQ_U64(kfdr_sysfs_gpu_id(), 222);
    add_render(144); /* two GPUs visible: ambiguous, no translation */
    CHECK_EQ_U64(kfdr_sysfs_gpu_id(), 0);
    rm_render(136);
    rm_render(144);
    CHECK_EQ_U64(kfdr_sysfs_gpu_id(), 0); /* none */
}

static void expect_xlate(const char *in, const char *want) {
    char buf[512];
    const char *got = kfdr_topo_xlate(in, buf, sizeof buf);
    if (strcmp(got, want)) {
        fprintf(stderr, "topo_xlate(%s) = %s, want %s\n", in, got, want);
        test_failures++;
    }
    if (!strcmp(in, want) && got != in) {
        fprintf(stderr, "topo_xlate(%s) copied an unchanged path\n", in);
        test_failures++;
    }
}

static void test_path_rewrite(void) {
    topo_old = 3;
    topo_new = 4;
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/3/properties", "/sys/class/kfd/kfd/topology/nodes/4/properties");
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/3/mem_banks/0/properties",
                 "/sys/class/kfd/kfd/topology/nodes/4/mem_banks/0/properties");
    expect_xlate("/sys/devices/virtual/kfd/kfd/topology/nodes/3", "/sys/devices/virtual/kfd/kfd/topology/nodes/4");
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/31/properties", "/sys/class/kfd/kfd/topology/nodes/31/properties");
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/3x/properties", "/sys/class/kfd/kfd/topology/nodes/3x/properties");
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/2/gpu_id", "/sys/class/kfd/kfd/topology/nodes/2/gpu_id");
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/", "/sys/class/kfd/kfd/topology/nodes/");
    expect_xlate("/sys/class/kfd/kfd/topology/system_properties", "/sys/class/kfd/kfd/topology/system_properties");
    expect_xlate("/etc/hosts", "/etc/hosts");

    topo_new = -1; /* not restored on another GPU */
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/3/properties", "/sys/class/kfd/kfd/topology/nodes/3/properties");
    topo_new = 3; /* same node */
    expect_xlate("/sys/class/kfd/kfd/topology/nodes/3/properties", "/sys/class/kfd/kfd/topology/nodes/3/properties");

    char buf[16];
    CHECK(kfdr_topo_xlate(NULL, buf, sizeof buf) == NULL);
}

int main(void) {
    const char *tmp = getenv("TMPDIR");
    snprintf(root, sizeof root, "%s/kfdr-topo-XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(root)) { perror("mkdtemp"); return 2; }
    snprintf(nodes, sizeof nodes, "%s/nodes", root);
    snprintf(dri, sizeof dri, "%s/dri", root);
    mkdir(nodes, 0755);
    mkdir(dri, 0755);
    kfdr_topology_dir = nodes;
    kfdr_dri_dir = dri;

    test_sysfs();
    test_path_rewrite();

    char cmd[600];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", root);
    if (system(cmd)) { /* best effort */ }
    TEST_DONE("test_topology");
}
