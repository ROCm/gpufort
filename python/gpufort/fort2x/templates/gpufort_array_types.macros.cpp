{########################################################################################}
{%- macro render_gpufort_array_c_bindings(thistype,max_rank) -%}
{# C side #}
{% for rank in range(1,max_rank+1) %}
{%   set rank_ub = rank+1 %}
{%   set c_type = "char" %}
{%   set c_prefix = "gpufort_array"+rank|string %}
{%   for async_suffix in ["_async",""] %}
{%     set is_async = async_suffix == "_async" %}
GPUFORT_HOST_ROUTINE gpufort::error {{c_prefix}}_init{{async_suffix}} (
  gpufort::array{{rank}}<{{c_type}}>* array,
  int bytes_per_element,
  {{c_type}}* data_host,
  {{c_type}}* data_dev,
  int* sizes,
  int* lbounds,{{"
  gpufort::queue stream," if is_async}}
  gpufort::AllocMode alloc_mode,
  gpufort::SyncMode sync_mode
) {
  return array->init{{async_suffix}}(
    bytes_per_element,
    data_host, data_dev,
    sizes, lbounds,
    {{"stream," if is_async}}alloc_mode, sync_mode
  );
} 

GPUFORT_HOST_ROUTINE gpufort::error {{c_prefix}}_destroy{{async_suffix}}(
  gpufort::array{{rank}}<{{c_type}}>* array{{",
  gpufort::queue stream" if is_async}}
) {
  return array->destroy{{async_suffix}}({{"stream" if is_async}});
} 

GPUFORT_HOST_ROUTINE gpufort::error {{c_prefix}}_copy_data_to_host{{async_suffix}} (
  gpufort::array{{rank}}<{{c_type}}>* array{{",
  gpufort::queue stream" if is_async}}
) {
  return array->copy_data_to_host{{async_suffix}}({{"stream" if is_async}});
} 

GPUFORT_HOST_ROUTINE gpufort::error {{c_prefix}}_copy_data_to_device{{async_suffix}} (
  gpufort::array{{rank}}<{{c_type}}>* array{{",
  gpufort::queue stream" if is_async}}
) {
  return array->copy_data_to_device{{async_suffix}}({{"stream" if is_async}});
} 
{%   endfor %}{# async_suffix #}
{% endfor %}{# rank #}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_dope_c_bindings(thistype,max_rank) -%}
{% for rank in range(1,max_rank+1) %}
{%   set rank_ub = rank+1 %}
{%   set c_prefix = "gpufort_dope"+rank|string %}
GPUFORT_HOST_ROUTINE gpufort::error {{c_prefix}}_init(
  gpufort::dope{{rank}}<{{c_type}}>* dope,
  int* sizes,
  int* lbounds) {
  return dope->init(sizes, lbounds);
} 
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_gpufort_array_ptr_c_bindings(thistype,max_rank) -%}
{% for rank in range(1,max_rank+1) %}
{%   set rank_ub = rank+1 %}
{%   set c_prefix = "gpufort_array_ptr"+rank|string %}
GPUFORT_HOST_ROUTINE gpufort::error {{c_prefix}}_init(
  gpufort::array_ptr{{rank}}<{{c_type}}>* array_ptr,
  int bytes_per_element,
  {{c_type}}* data,
  int* sizes,
  int* lbounds) {
  return array_ptr->init(bytes_per_element,data,sizes, lbounds);
} 
{% endfor %}
{%- endmacro -%}
{########################################################################################}
{%- macro render_array_property_inquiry_c_bindings(thistype,max_rank,parameterized=True) -%}
{# parameterized - If we have a template parameter to consider #}
{% set param = "<char>" if parameterized else "" %}
{% for rank in range(1,max_rank+1) %}
/**
 * \return size of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_HOST_ROUTINE  int gpufort_{{thistype}}{{rank}}_size(
  gpufort::array{{rank}}{{param}}* array,
  int dim) {
  return array->data_dev.size(dim);
}

/**
 * \return lower bound (inclusive) of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_HOST_ROUTINE  int gpufort_{{thistype}}{{rank}}_lbound(
  gpufort::array{{rank}}{{param}}* array,
  int dim) {
  return array->data_dev.lbound(dim);
}

/**
 * \return upper bound (inclusive) of the array in dimension 'dim'.
 * \param[in] dim selected dimension: 1,...,{{rank}}
 */
GPUFORT_HOST_ROUTINE  int gpufort_{{thistype}}{{rank}}_ubound(
  gpufort::array{{rank}}{{param}}* array,
  int dim) {
  return array->data_dev.ubound(dim);
}
{% endfor %}
{%- endmacro -%}  
{########################################################################################}
{%- macro render_memory_operations_c_bindings(thistype,rank,async_suffix) -%}  
GPUFORT_HOST_ROUTINE gpufort::error gpufort_{{thistype}}{{rank}}_copy_data_to_buffer{{async_suffix}} (
  gpufort::{{thistype}}{{rank}}<char>* array,
  void* buffer,
  gpufort::Direction direction{{",
  gpufort::queue stream" if is_async}}
) {
  return array->copy_data_to_buffer{{async_suffix}}(buffer,direction{{",stream" if is_async}});
} 

GPUFORT_HOST_ROUTINE gpufort::error gpufort_{{thistype}}{{rank}}_copy_data_from_buffer{{async_suffix}} (
  gpufort::{{thistype}}{{rank}}<char>* array,
  void* buffer,
  gpufort::Direction direction{{",
  gpufort::queue stream" if is_async}}
) {
  return array->copy_data_from_buffer{{async_suffix}}(buffer,direction{{",stream" if is_async}});
} 

{% for target in ["host","device"] %}
{% set is_host = target == "host" %}
/**
 * Allocate a {{target}} buffer with
 * the same size as the data buffers
 * associated with this gpufort array.
 * \see size_in_bytes()
 * \param[inout] pointer to the buffer to allocate
{% if is_host %}
 * \param[in] pinned If the memory should be pinned (default=true)
 * \param[in] flags  Flags for the host memory allocation (default=0).
{% endif %}
 */
GPUFORT_HOST_ROUTINE gpufort::error gpufort_{{thistype}}{{rank}}_allocate_{{target}}_buffer(
  gpufort::{{thistype}}{{rank}}<char>* array,
  void** buffer{{",bool pinned=true,int flags=0" if is_host}}
) {
  return array->allocate_{{target}}_buffer(buffer{{",pinned,flags" if is_host}});
}

/**
 * Deallocate a {{target}} buffer
 * created via the allocate_{{target}}_buffer routine.
 * \see size_in_bytes(), allocate_{{target}}_buffer
 * \param[inout] the buffer to deallocte
{% if is_host %}
 * \param[in] pinned If the memory to deallocate is pinned [default=true]
{% endif %}
 */
GPUFORT_HOST_ROUTINE gpufort::error gpufort_{{thistype}}{{rank}}_deallocate_{{target}}_buffer(
  gpufort::{{thistype}}{{rank}}<char>* array,
  void* buffer{{",bool pinned=true" if is_host}}) {
  return array->deallocate_{{target}}_buffer(buffer{{",pinned" if is_host}});
}
{% endfor %}{# target #}
{%- endmacro -%}
