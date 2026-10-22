/* siphon: GGUF parser tests.
 *
 * Builds synthetic GGUF files byte-by-byte (a minimal writer lives here
 * for test purposes), then verifies the parser against known values,
 * including all metadata value types, tensor geometry for several quant
 * formats, alignment handling, and malformed-input rejection.
 */
#include "siphon/gguf.h"
#include "test_util.h"

#include <sys/stat.h>
#include <unistd.h>

/* ---- tiny growable buffer used as a GGUF writer --------------------- */

typedef struct {
    uint8_t *d;
    size_t len, cap;
} wbuf;

static void wput(wbuf *w, const void *p, size_t n) {
    if (w->len + n > w->cap) {
        w->cap = (w->len + n) * 2 + 256;
        w->d = si_xrealloc(w->d, w->cap);
    }
    memcpy(w->d + w->len, p, n);
    w->len += n;
}
static void wu8(wbuf *w, uint8_t v)  { wput(w, &v, 1); }
static void wu32(wbuf *w, uint32_t v) { wput(w, &v, 4); }
static void wu64(wbuf *w, uint64_t v) { wput(w, &v, 8); }
static void wf32(wbuf *w, float v)    { wput(w, &v, 4); }
static void wstr(wbuf *w, const char *s) {
    wu64(w, strlen(s));
    wput(w, s, strlen(s));
}

/* ---- build a synthetic GGUF v3 into a file -------------------------- */

static char g_path[256];

static const char *write_gguf(wbuf *w, uint32_t version, uint64_t align) {
    snprintf(g_path, sizeof(g_path), "build/tmp-gguf-%u.gguf", version);
    FILE *f = fopen(g_path, "wb");
    if (!f) return NULL;
    fwrite(w->d, 1, w->len, f);
    fclose(f);
    (void) align;
    return g_path;
}

static void build_file(uint32_t version) {
    wbuf w = {0};

    /* header */
    wput(&w, "GGUF", 4);
    wu32(&w, version);
    wu64(&w, 3);   /* tensors */
    wu64(&w, 16);  /* kv pairs */

    /* metadata: one of each interesting type */
    wstr(&w, "test.u8");    wu32(&w, SI_GGUF_V_U8);  wu8(&w, 200);
    wstr(&w, "test.i8");    wu32(&w, SI_GGUF_V_I8);  wu8(&w, 0xC8);  /* -56 */
    wstr(&w, "test.u16");   wu32(&w, SI_GGUF_V_U16); uint16_t u16 = 60000; wput(&w, &u16, 2);
    wstr(&w, "test.i16");   wu32(&w, SI_GGUF_V_I16); int16_t i16 = -1234; wput(&w, &i16, 2);
    wstr(&w, "test.u32");   wu32(&w, SI_GGUF_V_U32); wu32(&w, 4000000000u);
    wstr(&w, "test.i32");   wu32(&w, SI_GGUF_V_I32); uint32_t i32 = (uint32_t) -77; wput(&w, &i32, 4);
    wstr(&w, "test.f32");   wu32(&w, SI_GGUF_V_F32); wf32(&w, 1.5f);
    wstr(&w, "test.bool");  wu32(&w, SI_GGUF_V_BOOL); wu8(&w, 1);
    wstr(&w, "test.u64");   wu32(&w, SI_GGUF_V_U64); wu64(&w, 9000000000ull);
    wstr(&w, "test.i64");   wu32(&w, SI_GGUF_V_I64); uint64_t i64 = (uint64_t) -5; wput(&w, &i64, 8);
    wstr(&w, "test.f64");   wu32(&w, SI_GGUF_V_F64); double d = -0.25; wput(&w, &d, 8);
    wstr(&w, "test.string"); wu32(&w, SI_GGUF_V_STRING); wstr(&w, "hello \xe2\x96\x81 world");
    /* arrays: u32[3], f32[2], string[2] */
    wstr(&w, "test.arr_u32");
    wu32(&w, SI_GGUF_V_ARRAY); wu32(&w, SI_GGUF_V_U32); wu64(&w, 3);
    wu32(&w, 10); wu32(&w, 20); wu32(&w, 30);
    wstr(&w, "test.arr_f32");
    wu32(&w, SI_GGUF_V_ARRAY); wu32(&w, SI_GGUF_V_F32); wu64(&w, 2);
    wf32(&w, 2.5f); wf32(&w, -2.5f);
    wstr(&w, "test.arr_str");
    wu32(&w, SI_GGUF_V_ARRAY); wu32(&w, SI_GGUF_V_STRING); wu64(&w, 2);
    wstr(&w, "alpha"); wstr(&w, "beta");
    /* general.alignment */
    wstr(&w, "general.alignment"); wu32(&w, SI_GGUF_V_U32); wu32(&w, 32);

    /* tensor infos:
     *   t_f32:  [4, 3]      F32   -> 48 bytes
     *   t_q40:  [64]        Q4_0  -> 2 blocks * 18 = 36 bytes
     *   t_q6k:  [256, 2]    Q6_K  -> 2 blocks * 210 = 420 bytes
     * Offsets chosen aligned to 32, data laid out contiguously. */
    wstr(&w, "t_f32");
    wu32(&w, 2); wu64(&w, 4); wu64(&w, 3);
    wu32(&w, 0 /*F32*/); wu64(&w, 0);
    wstr(&w, "t_q40");
    wu32(&w, 1); wu64(&w, 64);
    wu32(&w, 2 /*Q4_0*/); wu64(&w, 64);       /* aligned after 48 */
    wstr(&w, "t_q6k");
    wu32(&w, 2); wu64(&w, 256); wu64(&w, 2);
    wu32(&w, 14 /*Q6_K*/); wu64(&w, 128);     /* 64 + 36 = 100 -> 128 */

    /* pad to 32, then data */
    while (w.len % 32 != 0) wu8(&w, 0);

    /* t_f32 data: 12 floats, value = index */
    for (int i = 0; i < 12; i++) wf32(&w, (float) i);
    /* pad to 64 */
    while (w.len % 32 != 0) wu8(&w, 0);
    /* t_q40 data: 36 bytes with a marker pattern */
    for (int i = 0; i < 36; i++) wu8(&w, (uint8_t) (0xA0 + i));
    /* pad to 128 */
    while (w.len % 32 != 0) wu8(&w, 0);
    /* t_q6k data: 420 bytes */
    for (int i = 0; i < 420; i++) wu8(&w, (uint8_t) (0x30 + (i % 100)));

    write_gguf(&w, version, 32);
    free(w.d);
}

static void test_roundtrip(void) {
    for (int v = 2; v <= 3; v++) {
        build_file((uint32_t) v);
        si_gguf *g = NULL;
        int rc = si_gguf_open(g_path, &g);
        CHECK(rc == SI_OK);
        if (rc != SI_OK) { si_log_err("%s", si_errstr()); continue; }

        CHECK(g->version == (uint32_t) v);
        CHECK(g->n_tensors == 3);
        CHECK(g->n_kv == 16);
        CHECK(g->alignment == 32);

        CHECK(si_gguf_get_u64(g, "test.u8", 0) == 200);
        CHECK(si_gguf_get_i64(g, "test.i8", 0) == -56);
        CHECK(si_gguf_get_u64(g, "test.u16", 0) == 60000);
        CHECK(si_gguf_get_i64(g, "test.i16", 0) == -1234);
        CHECK(si_gguf_get_u64(g, "test.u32", 0) == 4000000000ull);
        CHECK(si_gguf_get_i64(g, "test.i32", 0) == -77);
        CHECK(si_gguf_get_f64(g, "test.f32", 0) == 1.5);
        CHECK(si_gguf_get_bool(g, "test.bool", false) == true);
        CHECK(si_gguf_get_u64(g, "test.u64", 0) == 9000000000ull);
        CHECK(si_gguf_get_i64(g, "test.i64", 0) == -5);
        CHECK(si_gguf_get_f64(g, "test.f64", 0) == -0.25);

        si_gguf_str s = si_gguf_get_str(g, "test.string");
        CHECK(s.len == 15);
        CHECK(memcmp(s.ptr, "hello \xe2\x96\x81 world", 15) == 0);

        const si_gguf_kv *a = si_gguf_find_arr(g, "test.arr_u32");
        CHECK(a != NULL && a->v.arr.count == 3);
        if (a) {
            const uint32_t *p = (const uint32_t *) a->v.arr.data;
            CHECK(p[0] == 10 && p[1] == 20 && p[2] == 30);
        }
        const si_gguf_kv *af = si_gguf_find_arr(g, "test.arr_f32");
        CHECK(af != NULL && af->v.arr.count == 2);
        if (af) {
            const float *p = (const float *) af->v.arr.data;
            CHECK(p[0] == 2.5f && p[1] == -2.5f);
        }
        si_gguf_str e0, e1;
        CHECK(si_gguf_arr_str_get(g, "test.arr_str", 0, &e0) == SI_OK);
        CHECK(si_gguf_arr_str_get(g, "test.arr_str", 1, &e1) == SI_OK);
        CHECK(e0.len == 5 && memcmp(e0.ptr, "alpha", 5) == 0);
        CHECK(e1.len == 4 && memcmp(e1.ptr, "beta", 4) == 0);
        CHECK(si_gguf_arr_str_get(g, "test.arr_str", 2, &e0) == SI_ERANGE);

        /* tensors */
        const si_gguf_tensor *t = si_gguf_tensor_find(g, "t_f32");
        CHECK(t != NULL);
        if (t) {
            CHECK(t->n_dims == 2 && t->dims[0] == 4 && t->dims[1] == 3);
            CHECK(t->ggml_type == 0);
            CHECK(t->n_elems == 12);
            CHECK(t->n_bytes == 48);
            CHECK(t->offset == 0);
            const float *fd = (const float *) t->data;
            CHECK(fd[0] == 0.f && fd[11] == 11.f);
        }
        t = si_gguf_tensor_find(g, "t_q40");
        CHECK(t != NULL);
        if (t) {
            CHECK(t->n_elems == 64 && t->n_bytes == 36 && t->offset == 64);
            CHECK(t->data[0] == 0xA0 && t->data[35] == 0xC3);
        }
        t = si_gguf_tensor_find(g, "t_q6k");
        CHECK(t != NULL);
        if (t) {
            CHECK(t->n_elems == 512 && t->n_bytes == 420 && t->offset == 128);
            CHECK(t->data[0] == 0x30 && t->data[419] == (uint8_t) (0x30 + (419 % 100)));
        }
        CHECK(si_gguf_tensor_find(g, "nope") == NULL);

        /* type geometry sanity: K-quant bpw values from ggml-common.h */
        CHECK(si_gguf_type_info(12).type_size == 144);   /* Q4_K */
        CHECK(si_gguf_type_info(11).type_size == 110);   /* Q3_K */
        CHECK(si_gguf_type_info(10).type_size == 84);    /* Q2_K */
        CHECK(si_gguf_type_info(14).type_size == 210);   /* Q6_K */
        CHECK(si_gguf_type_info(8).type_size == 34);     /* Q8_0 */
        CHECK(si_gguf_type_info(1).blck_size == 1);      /* F16 */
        CHECK(!si_gguf_type_info(4).known);              /* removed Q4_2 */

        si_gguf_close(g);
        unlink(g_path);
    }
}

static void test_bad_files(void) {
    wbuf w = {0};
    wput(&w, "GGUF", 4); wu32(&w, 3); wu64(&w, 0); wu64(&w, 1);
    wstr(&w, "k"); wu32(&w, SI_GGUF_V_U8); wu8(&w, 1);
    write_gguf(&w, 3, 32);

    /* bad magic */
    {
        wbuf b = {0};
        wput(&b, "GUFF", 4); wu32(&b, 3); wu64(&b, 0); wu64(&b, 0);
        write_gguf(&b, 3, 32);
        si_gguf *g = NULL;
        CHECK(si_gguf_open(g_path, &g) == SI_EFORMAT);
        CHECK(g == NULL);
        free(b.d);
    }
    /* bad version */
    {
        wbuf b = {0};
        wput(&b, "GGUF", 4); wu32(&b, 1); wu64(&b, 0); wu64(&b, 0);
        write_gguf(&b, 1, 32);
        si_gguf *g = NULL;
        CHECK(si_gguf_open(g_path, &g) == SI_ENOTSUP);
        free(b.d);
    }
    /* truncated mid-kv */
    {
        si_gguf *g = NULL;
        FILE *f = fopen(g_path, "rb");
        CHECK(f != NULL);
        if (f) {
            uint8_t head[40];
            size_t n = fread(head, 1, sizeof(head), f);
            fclose(f);
            FILE *o = fopen(g_path, "wb");
            fwrite(head, 1, n < 30 ? n : 30, o);
            fclose(o);
            CHECK(si_gguf_open(g_path, &g) != SI_OK);
        }
    }
    /* tensor out of bounds */
    {
        wbuf b = {0};
        wput(&b, "GGUF", 4); wu32(&b, 3); wu64(&b, 1); wu64(&b, 0);
        wstr(&b, "t"); wu32(&b, 1); wu64(&b, 32); wu32(&b, 0); wu64(&b, 0);
        while (b.len % 32) wu8(&b, 0);
        for (int i = 0; i < 16; i++) wf32(&b, 0.f);
        write_gguf(&b, 3, 32);
        si_gguf *g = NULL;
        CHECK(si_gguf_open(g_path, &g) == SI_EFORMAT); /* 32 floats missing 64 */
        free(b.d);
    }
    /* nonexistent path */
    {
        si_gguf *g = NULL;
        CHECK(si_gguf_open("build/does-not-exist.gguf", &g) == SI_ENOENT);
    }

    free(w.d);
    unlink(g_path);
}

int main(void) {
    mkdir("build", 0777);
    test_roundtrip();
    test_bad_files();
    TEST_REPORT("test_gguf");
}
