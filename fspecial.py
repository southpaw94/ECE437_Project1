import numpy as np

def fspecial(filt_type, **kwargs):
	"""
	Python implementation of fspecial function
	"""

	if filt_type == 'average' or filt_type == 'avg':
		fsize = kwargs.pop('fsize', 3)

		f = np.ones(fsize)
		f /= f.sum()

		return f

	elif filt_type == 'disk' or filt_type == 'disc':
		r = kwargs.pop('radius', 5)

		if r == 0:
			return 1
		else:
			ax = r
			corner = int(r / np.sqrt(2) + 0.5) - 0.5
			rsq = r ** 2

			X, Y = np.meshgrid(np.linspace(-r, r, 2 * r + 1), 
								 np.linspace(-r, r, 2 * r + 1))
			
			rhi = (np.abs(X) + 0.5) ** 2 + (np.abs(Y) + 0.5) ** 2
			f = (rhi <= rsq) / 1.0
			xx = np.linspace(0.5, r - 0.5, r)
			ii = np.sqrt(rsq - xx ** 2)
			tmp = np.sqrt(rsq - 0.25)
			rint = (0.5 * tmp + rsq * np.arctan(0.5 / tmp)) / 2
			cap = 2 * rint - r + 0.5
			f[ax, ax + r] = cap
			f[ax, ax - r] = cap
			f[ax + r, ax] = cap
			f[ax - r, ax] = cap

			if r == 1:
				y = ii[1]
				lint = rint
				tmp = np.sqrt(rsq - y ** 2)
				rint = (y * tmp + rsq * arctan (y / tmp)) / 2
				val = rint - lint - 0.5 * (y - 0.5)
				f[ax - r, ax -r ] = val
				f[ax+r, ax-r] = val
				f[ax-r, ax+r] = val
				f[ax+r, ax+r] = val
			else:
				idx = 1
				x = 0.5
				y = r - 0.5
				rx = 0.5
				ybreak = False
				
				while True:
					i = x + 0.5
					j = y + 0.5
					lint = rint
					lx = rx
					if ybreak == True:
						ybreak = False
						val = lx - x
						idx += 1
						x += 1
						rx = x
						val -= y * (x - lx)
					elif ii[idx] < y:
						ybreak = True
						y -= 1
						rx = ii[int(y + 0.5)]
						val = (y + 1) * (x - rx)
					else:
						val = -y
						idx += 1
						x += 1
						rx = x
						if int(ii[idx] - 0.5) == y:
							y += 1

					tmp = np.sqrt(rsq - rx ** 2)
					rint = (rx * tmp + rsq * np.arctan(rx / tmp)) / 2
					val += rint - lint
					f[int(ax+i), int(ax+j)] = val
					f[int(ax+i), int(ax-j)] = val
					f[int(ax-i), int(ax+j)] = val
					f[int(ax-i), int(ax-j)] = val
					f[int(ax+j), int(ax+i)] = val
					f[int(ax+j), int(ax-i)] = val
					f[int(ax-j), int(ax+i)] = val
					f[int(ax-j), int(ax+i)] = val
					f[int(ax-j), int(ax-i)] = val

					if y < corner or x > corner:
						break
			f /= np.pi * rsq
			return f

	elif filt_type=='gaussian':
		"""
		fspecial('gaussian', lengths=(3, 3), sigma=0.5)
		"""
		shape = kwargs.pop('shape', (3, 3))
		sigma = kwargs.pop('sigma', (0.5))
		
		m, n = [(ss - 1.)/2. for ss in shape]
		y, x = np.ogrid[-m:m+1, -n:n+1]
		h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
		h[h < np.finfo(h.dtype).eps*h.max()] = 0
		sumh = h.sum()
		if sumh != 0:
			h /= sumh
		return h

	elif filt_type=='unsharp':
		alpha = kwargs.pop('alpha', 0.2)

		f = (1 / (alpha + 1)) * np.array(([-alpha, alpha-1, -alpha],
										 [alpha-1, alpha+5, alpha-1],
										 [-alpha, alpha-1, -alpha]))
		return f
