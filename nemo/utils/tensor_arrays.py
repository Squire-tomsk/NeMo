import math
import typing
import torch
import collections


class DictArray:
	def __init__(self, dicts, types = {}, *, batch_size: int = 1024, string_encoding = 'utf_16_le', intlist_encoding = torch.int64):
		dicts = list(dicts)
		numel = len(dicts)
		assert numel > 0
		self.none_keys = [key for key, value in types.items() if value is None]
		types = {key: value for key, value in types.items() if value is not None}
		self.tensors = {k: t(numel) for k, t in types.items() if t != StringArray and t != IntListArray}
		string_lists = {k: [None] * numel for k, t in types.items() if t == StringArray}
		intlist_lists = {k: [None] * numel for k, t in types.items() if t == IntListArray}
		temp_lists = {k: [None] * batch_size for k in self.tensors}
		for b in range(math.ceil(numel / batch_size)):
			for i, t in enumerate(dicts[b * batch_size: (b + 1) * batch_size]):
				for k in temp_lists:
					temp_lists[k][i] = t[k]
				for k in string_lists:
					string_lists[k][b * batch_size + i] = t[k]
				for k in intlist_lists:
					intlist_lists[k][b * batch_size + i] = t[k]
			for k, v in temp_lists.items():
				res = self.tensors[k][b * batch_size: (b + 1) * batch_size]
				res.copy_(torch.as_tensor(v[:len(res)], dtype=self.tensors[k].dtype))
		self.string_arrays = {k: StringArray(v, encoding=string_encoding) for k, v in string_lists.items()}
		self.intlist_arrays = {k: IntListArray(v, dtype=intlist_encoding) for k, v in intlist_lists.items()}

	def __getitem__(self, i):
		return dict(**{k: v[i].item() for k, v in self.tensors.items()}, **{k: v[i] for k, v in self.string_arrays.items()}, **{k: v[i] for k, v in self.intlist_arrays.items()}, **{k: None for k in self.none_keys})

	def __len__(self):
		return len(next(iter(self.tensors.values()))) if len(self.tensors) > 0 else len(next(iter(self.string_arrays.values())))


class NamedTupleArray(DictArray):
	def __init__(self, tuples, types = {}, **kwargs):
		self.type = type(tuples[0])
		dicts = [t._asdict() for t in tuples]
		super().__init__(dicts, types, **kwargs)

	def __getitem__(self, i):
		return self.type(**super().__getitem__(i))


class IntListArray:
	def __init__(self, intlists, dtype=torch.int64):
		tensors = [t.to(dtype=dtype) if torch.is_tensor(t) else torch.as_tensor(t, dtype=dtype) for t in intlists]
		self.data = torch.cat(tensors)
		self.cumlen = torch.LongTensor(list(map(len, tensors))).cumsum(dim=0)

	def __getitem__(self, i):
		return self.data[(self.cumlen[i - 1] if i >= 1 else 0): self.cumlen[i]].tolist()

	def __len__(self):
		return len(self.cumlen)


class StringArray:
	def __init__(self, strings, encoding = 'utf_16_le'):
		strings = list(strings)
		self.encoding = encoding
		self.multiplier = dict(ascii=1, utf_16_le=2, utf_32_le=4)[encoding]
		self.data = torch.ByteTensor(torch.ByteStorage.from_buffer(''.join(strings).encode(encoding)))
		self.cumlen = torch.LongTensor(list(map(len, strings))).cumsum(dim=0)
		assert int(self.cumlen[-1]) * self.multiplier == len(self.data), f'[{encoding}] is not enough to hold characters, use a larger character class'

	def __getitem__(self, i):
		return bytes(self.data[(self.cumlen[i - 1] * self.multiplier if i >= 1 else 0): self.cumlen[i] * self.multiplier]).decode(self.encoding)

	def __len__(self):
		return len(self.cumlen)
