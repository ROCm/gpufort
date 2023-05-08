
# derived from https://www.openacc.org/sites/default/files/inline-images/Specification/OpenACC-3.1-final.pdf (07/25/2022)
acc_clauses = {
  "parallel": [
    "async",
    "wait",
    "num_gangs",
    "num_workers",
    "vector_length",
    "device_type",
    "if",
    "self",
    "reduction",
    "copy",
    "copyin",
    "copyout",
    "create",
    "no_create",
    "present",
    "deviceptr",
    "attach",
    "private",
    "firstprivate",
    "default",
  ],
  "data": [
    "if",
    "copy",
    "copyin",
    "copyout",
    "create",
    "no_create",
    "present",
    "deviceptr",
    "attach",
    "default",
  ],
  "enter data": [
    "if",
    "async",
    "wait",
    "copyin",
    "create",
    "attach",
  ],
  "exit data": [
    "if",
    "async",
    "wait",
    "copyout",
    "delete",
    "detach",
    "finalize",
  ],
  "host_data": [
    "use_device",
    "if",
    "if_present",
  ],
  "loop": [
    "collapse",
    "gang",
    "worker",
    "vector",
    "seq",
    "independen",
    "auto",
    "tile",
    "device_type",
    "private",
    "reduction",
  ],
  "cache": [
  ],
  "atomic": [
    "read",
    "write",
    "update",
    "capture",
  ],
  "declare": [
    "copy",
    "copyin",
    "copyout",
    "create",
    "present",
    "deviceptr",
    "device_resident",
    "link",
  ],
  "routine": [
    "gang",
    "worker",
    "vector",
    "seq",
    "bind",
    "bind",
    "device_type",
    "nohost",
  ],
  "init": [
    "device_type",
    "device_num",
    "if",
  ],
  "set": [
    "device_type",
    "device_num",
    "if",
  ],
  "update": [
    "async",
    "wait",
    "device_type",
    "if",
    "if_present",
    "self",
    "host",
    "device",
  ],
  "wait": [
    "async",
    "if",
  ],
}

acc_clauses["serial"] = list(acc_clauses["parallel"])
acc_clauses["serial"].remove("num_gangs")
acc_clauses["serial"].remove("num_workers")
acc_clauses["serial"].remove("vector_length")

acc_clauses["kernels"] = list(acc_clauses["parallel"])
acc_clauses["kernels"].remove("reduction")
acc_clauses["kernels"].remove("private")
acc_clauses["kernels"].remove("firstprivate")

acc_clauses["parallel loop"] = acc_clauses["parallel"] + acc_clauses["loop"]
acc_clauses["serial loop"] = acc_clauses["serial"] + acc_clauses["loop"]
acc_clauses["kernels loop"] = acc_clauses["kernels"] + acc_clauses["loop"]

acc_clauses["shutdown"] = list(acc_clauses["init"])

acc_clauses["set"] = list(acc_clauses["init"])
acc_clauses["set"].append("default_async")
