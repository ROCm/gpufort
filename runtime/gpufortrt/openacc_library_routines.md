---
geometry: margin=2cm
---

# Implemented API

| API | Lang\* | OpenACC | GPUFORTRT\*\* | Priority\*\*\* | 
|-----|--------|---------|-------------|----------|
|acc\_get\_num\_devices|C/C++, Fortran|implemented|implemented|high|
|acc\_set\_device\_type|C/C++, Fortran|implemented|implemented|high|
|acc\_get\_device\_type|C/C++, Fortran|implemented|implemented|high|
|acc\_set\_device\_num|C/C++, Fortran|implemented|implemented||
|acc\_get\_device\_num|C/C++, Fortran|implemented|implemented||
|acc\_get\_property|C/C++, Fortran|implemented|implemented||
|acc\_init|C/C++, Fortran|implemented|implemented||
|acc\_shutdown|C/C++, Fortran|implemented|implemented||
|acc\_async\_test|C/C++, Fortran|implemented|implemented||
|acc\_async\_test\_device|C/C++, Fortran|implemented|implemented||
|acc\_async\_test\_all|C/C++, Fortran|implemented|implemented||
|acc\_async\_test\_all\_device|C/C++, Fortran|implemented|implemented||
|acc\_wait|C/C++, Fortran|implemented|implemented||
|acc\_wait\_device|C/C++, Fortran|implemented|implemented|high|
|acc\_wait\_async|C/C++, Fortran|implemented|implemented||
|acc\_wait\_device\_async|C/C++, Fortran|implemented|implemented|high|
|acc\_wait\_all|C/C++, Fortran|implemented|implemented||
|acc\_wait\_all\_device|C/C++, Fortran|implemented|implemented|high|
|acc\_wait\_all\_async|C/C++, Fortran|implemented|implemented||
|acc\_wait\_all\_device\_async|C/C++, Fortran|implemented|implemented|high|
|acc\_get\_default\_async|C/C++, Fortran|implemented|implemented||
|acc\_set\_default\_async|C/C++, Fortran|implemented|implemented||
|acc\_on\_device||||low|
|acc\_malloc||||low|
|acc\_free||||low|
|acc\_copyin|C/C++, Fortran|implemented|implemented||
|acc\_create|C/C++, Fortran|implemented|implemented||
|acc\_copyout|C/C++, Fortran|implemented|implemented||
|acc\_delete|C/C++, Fortran|implemented|implemented||
|acc\_update\_device|C/C++, Fortran|implemented|implemented||
|acc\_update\_self|C/C++, Fortran|implemented|implemented||
|acc\_map\_data||||low|
|acc\_unmap\_data||||low|
|acc\_deviceptr|C/C++||implemented||
|acc\_hostptr|C/C++|||low|
|acc\_is\_present|||implemented||
|acc\_memcpy\_to\_device||||low|
|acc\_memcpy\_from\_device||||low|
|acc\_memcpy\_device||||low|
|acc\_attach||||low|
|acc\_detach||||low|
|acc\_memcpy\_d2d||||low|

Remarks:

* \* While some APIs are exposed only to C according to the OpenACC standard, `GPUFORTRT` may expose some C interfaces also to Fortran. An \* indicates that this feature was exposed by the GPUFORTRT to Fortran despite the OpenACC standard not requiring this.
* \*\* `GPUFORTRT` signatures are prefixd by `gpufortrt_` instead of `acc_` and the number and meaning of 
arguments may differ compared to the OpenACC signature.
* \*\*\* Current priorities for implementing missing APIs. This column will disappear as soon as all are implemented.
