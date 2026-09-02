import torch


def circulant_plan(L1, L2, device, dtype=torch.float32):
    N = L1 * L2
    r = torch.arange(N, device=device) // L2
    c = torch.arange(N, device=device) % L2
    didx = ((r.unsqueeze(1) - r.unsqueeze(0)) % L1) * L2 + (c.unsqueeze(1) - c.unsqueeze(0)) % L2
    th1 = 2.0 * torch.pi * torch.outer(torch.arange(L1, device=device),
                                       torch.arange(L1, device=device)).to(dtype) / L1
    th2 = 2.0 * torch.pi * torch.outer(torch.arange(L2, device=device),
                                       torch.arange(L2, device=device)).to(dtype) / L2
    return {
        'L1': L1, 'L2': L2, 'N': N, 'didx': didx,
        'W1f_re': torch.cos(th1), 'W1f_im': -torch.sin(th1),
        'W2f_re': torch.cos(th2), 'W2f_im': -torch.sin(th2),
        'W1i_re': torch.cos(th1) / L1, 'W1i_im': torch.sin(th1) / L1,
        'W2i_re': torch.cos(th2) / L2, 'W2i_im': torch.sin(th2) / L2,
    }


def circulant_dense(Uk_re, Uk_im, plan):
    L1, L2, N = plan['L1'], plan['L2'], plan['N']
    A = Uk_re.shape[-1]
    Uk = torch.complex(Uk_re, Uk_im)
    Adisp = torch.fft.ifft2(Uk, dim=(0, 1)).reshape(N, A, A)
    M = Adisp[plan['didx']]
    M = M.permute(2, 0, 3, 1).reshape(A * N, A * N)
    return M.real.contiguous(), M.imag.contiguous()


def _axis_dft(vr, vi, Wre, Wim, spec):
    tr = torch.einsum(spec, vr, Wre) - torch.einsum(spec, vi, Wim)
    ti = torch.einsum(spec, vr, Wim) + torch.einsum(spec, vi, Wre)
    return tr, ti


def circulant_apply_k(re, im, Uk_re, Uk_im, plan):
    B = re.shape[0]
    L1, L2, N = plan['L1'], plan['L2'], plan['N']
    A = Uk_re.shape[-1]
    vr = re.view(B, A, L1, L2)
    vi = im.view(B, A, L1, L2)
    tr, ti = _axis_dft(vr, vi, plan['W2f_re'], plan['W2f_im'], 'baxy,ky->baxk')
    ur, ui = _axis_dft(tr, ti, plan['W1f_re'], plan['W1f_im'], 'bayk,jy->bajk')
    wr = torch.einsum('bijk,jkoi->bojk', ur, Uk_re) - torch.einsum('bijk,jkoi->bojk', ui, Uk_im)
    wi = torch.einsum('bijk,jkoi->bojk', ur, Uk_im) + torch.einsum('bijk,jkoi->bojk', ui, Uk_re)
    tr, ti = _axis_dft(wr, wi, plan['W1i_re'], plan['W1i_im'], 'bojk,xj->boxk')
    outr, outi = _axis_dft(tr, ti, plan['W2i_re'], plan['W2i_im'], 'boxk,yk->boxy')
    return outr.reshape(B, A * N), outi.reshape(B, A * N)
