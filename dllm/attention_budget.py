"""Bound the FP32 partial/LSE storage and the temporary chunk output per launch."""


def query_rows(heads,dim,splits,scratch_bytes,output_bytes=4):
    if type(scratch_bytes) is not int or scratch_bytes<=0:
        raise ValueError('attention scratch budget must be a positive byte count')
    per_row=heads*(splits*(dim+1)*4+dim*output_bytes)
    if scratch_bytes<per_row:
        raise ValueError('attention scratch budget cannot fit one query row')
    return scratch_bytes//per_row
