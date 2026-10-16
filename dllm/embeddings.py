"""Embedding request validation, independent of any model or optional dependency."""


def options(inputs,encoding_format='float',dimensions=None):
    if encoding_format not in ('float','base64'):
        raise ValueError('encoding_format must be float or base64')
    if dimensions is not None and (type(dimensions) is not int or dimensions <= 0):
        raise ValueError('dimensions must be a positive integer')
    if isinstance(inputs,str):
        inputs=[inputs]
    elif isinstance(inputs,list) and inputs and all(type(t) is int for t in inputs):
        inputs=[inputs]
    if not isinstance(inputs,list) or not 1 <= len(inputs) <= 64:
        raise ValueError('embeddings require 1..64 inputs')
    if all(isinstance(value,str) and value for value in inputs):
        try:
            for value in inputs:
                value.encode('utf-8')
        except UnicodeError as error:
            raise ValueError('embedding text must be valid UTF-8') from error
        return list(inputs),encoding_format,dimensions
    if all(isinstance(row,list) and row and all(type(t) is int and t>=0 for t in row) for row in inputs):
        return [list(row) for row in inputs],encoding_format,dimensions
    raise ValueError('input must be nonempty text, texts, token IDs or token-ID sequences')
