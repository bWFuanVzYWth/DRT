"""Regenerate OpenDRT/LPM golden outputs from the pinned upstream sources.

Run from the repository root: python references/validation/generate_hdr_vectors.py
Requires clang++ with native ext_vector_type support. Generated C++/executables
remain in ignored third_party/reference_sources; only reference vectors are tracked.
The actual upstream algorithms are compiled, not reimplemented in Python.
"""
from pathlib import Path
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "third_party/reference_sources"
COMPILER = shutil.which("clang++") or "C:/Program Files/LLVM/bin/clang++.exe"
INPUTS = "{{0,0,0},{.18f,.18f,.18f},{1,1,1},{16,16,16},{1,0,0},{0,1,0},{0,0,1},{.1f,.4f,4},{-.1f,.2f,.1f},{100,10,1},{4096,4096,4096}}"

ODRT = r'''
#include <cmath>
#include <cstdio>
#include <initializer_list>
#ifdef _WIN32
#include <windows.h>
#endif
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float2 __attribute__((ext_vector_type(2)));
inline float3 make_float3(float a,float b,float c){return {a,b,c};}
inline float2 make_float2(float a,float b){return {a,b};}
#define __DEVICE__
#define __CONSTANT__ static const
#define DEFINE_UI_PARAMS(n,label,t,d,...) static float n=d;
#define _powf powf
#define _sqrtf sqrtf
#define _fmaxf fmaxf
#define _fminf fminf
#define _expf expf
#define _exp2f exp2f
#define _logf logf
#define _log2f log2f
#define _fmod fmodf
#define _atan2f atan2f
#define _fabs fabsf
inline float _exp10f(float x){return powf(10.f,x);}
#include "display-transforms/opendrt/OpenDRT.dctl"
int main(){
#ifdef _WIN32
 SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
 in_gamut=1;in_oetf=0;display_encoding_preset=0;
 float inputs[][3]=INPUTS;
 for(float h:{1.f,4.f,10.f}){tn_Lp=100.f*h;for(auto& c:inputs){
  auto v=transform(1,1,0,0,c[0],c[1],c[2]);
  printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",h,c[0],c[1],c[2],powf(v.x,2.4f)*h,powf(v.y,2.4f)*h,powf(v.z,2.4f)*h);
 }}
}
'''

LPM_HEADER = r'''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif
#define A_CPU 1
#include "ffx-lpm/ffx_a.h"
static uint32_t ctl[24*4]{};
static void LpmSetupOut(AU1 i,inAU4 v){for(int j=0;j<4;j++)ctl[i*4+j]=v[j];}
#include "ffx-lpm/ffx_lpm.h"
struct V3 {float r,g,b;}; struct V2 {float x,y;};
float AMax3F1(float a,float b,float c){return fmaxf(a,fmaxf(b,c));}
using std::min;using std::max;
'''

LPM_MAIN = r'''
int main(){
#ifdef _WIN32
 SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
 float sat[3]={0,0,0},cross[3]={1,.5f,1.f/32.f};
 LpmSetup(false,LPM_CONFIG_709_2020,LPM_COLORS_709_2020,1.f/32.f,16.f,4.f,0.f,1.f,sat,cross);
 float f[96];memcpy(f,ctl,sizeof(f));
 float inputs[][3]=INPUTS;
 for(float h:{1.f,4.f,10.f})for(auto& x:inputs){
 // This is the workbench input adapter, outside the original LPM body.
 float r=x[0]*2.5214008886f+x[1]*-1.1339957494f+x[2]*-.3875618568f;
 float g=x[0]*-.2762140616f+x[1]*1.3725955663f+x[2]*-.0962823557f;
 float b=x[0]*-.0153202001f+x[1]*-.1529925618f+x[2]*1.1683871996f;
 float R=fmaxf(0.f,r*.627403895934699f+g*.329283038377884f+b*.043313065687417f);
 float G=fmaxf(0.f,r*.069097289358232f+g*.919540395075458f+b*.011362315566310f);
 float B=fmaxf(0.f,r*.016391438875150f+g*.088013307877226f+b*.895595253247624f);
 LpmMap(R,G,B,{f[25],f[26],f[27]},{f[6],f[7],f[8]},{f[12],f[13],f[14]},{f[0],f[1],f[2]},f[3],false,f[24],{f[4],f[5]},{f[9],f[10],f[11]},true,{f[30],f[31],f[32]},{f[33],f[34],f[35]},{f[36],f[37],f[38]},true,{f[28],f[29]},false,false,false,{},{},{});
 printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",h,x[0],x[1],x[2],R*h,G*h,B*h);
 }
}
'''


def extract_lpm_map(source):
    body = source[source.index(" void LpmMap("):]
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
    start = body.index("{")
    depth, end = 1, start + 1
    while depth:
        depth += (body[end] == "{") - (body[end] == "}")
        end += 1
    body = body[:end]
    return (body.replace("inout AF1", "float&").replace("AF3 ", "V3 ")
            .replace("AF2 ", "V2 ").replace("AF1 ", "float ").replace("AP1 ", "bool "))


records = []
for algorithm, commit in [
    ("opendrt", "af683323e2a8a63501f02c0a724ec538e3228ad0"),
    ("fidelityfx_lpm", "ed6ecd5b8963d2ec24603809b90ecfa00a1c3614"),
]:
    directory = CACHE / algorithm
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"{algorithm}: expected {commit}, found {actual}")
    source = ODRT if algorithm == "opendrt" else LPM_HEADER + extract_lpm_map((directory / "ffx-lpm/ffx_lpm.h").read_text()) + LPM_MAIN
    cpp = directory / "validate_reference.cpp"
    executable = directory / "validate_reference.exe"
    cpp.write_text(source.replace("INPUTS", INPUTS))
    subprocess.run([COMPILER, "-std=c++17", str(cpp), "-o", str(executable)], check=True, timeout=60)
    lines = subprocess.check_output([str(executable)], text=True, timeout=30).splitlines()
    for headroom in (1.0, 4.0, 10.0):
        vectors = []
        for line in lines:
            values = [float(v) for v in line.split()]
            if values[0] == headroom:
                vectors.append({"input_ap0": values[1:4], "expected_linear_rec709": values[4:7]})
        records.append({"algorithm": algorithm, "commit": commit, "headroom": headroom,
                        "input_space": "scene-linear AP0 D60", "output_space": "display-linear Rec709, SDR white = 1", "vectors": vectors})

destination = Path(__file__).with_name("hdr_reference_vectors.json")
destination.write_text(json.dumps({"format_version": 1, "records": records}, indent=2) + "\n")
print(f"Wrote {sum(len(record['vectors']) for record in records)} compiled-upstream vectors to {destination}")
