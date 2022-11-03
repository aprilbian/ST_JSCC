import numpy as np
from modules import *
from utils import complex_sig, pwr_normalize

# define end2end JSCC models 
class Mul_model(nn.Module):
    def __init__(self, args, enc, dec, res = None):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.P1 = args.P1
        self.P2 = args.P2
        self.P1_rng = args.P1_rng
        self.P2_rng = args.P2_rng
        self.Nr = args.Nr
        self.args = args

        self.enc = enc                      # Enc User B
        self.dec = dec                      # Source decoder
        self.res = res                      # Residual

    def MMSE_Equ(self, y, H, P):
        # MMSE Equ
        B = y.shape[0]
        P = P.unsqueeze(0).repeat(B,1,1).to(y.device)  # (B, 2, 2)
        HP = torch.bmm(P, H.permute(0,2,1).conj())
        RHH = torch.bmm(H, HP)                       # (B, Nt, Nt)
        eye = torch.eye(self.Nr, dtype=torch.cfloat).to(y.device).repeat(B, 1, 1)
        inv_Ryy = torch.inverse(RHH + eye)
        G = torch.bmm(HP, inv_Ryy)                   # (B, 2, Nr)
        x_equ = torch.bmm(G, y)
        return x_equ

    def RES_Equ(self, Y, H):
        # Residual Equ
        B, F = Y.shape[0], Y.shape[2]                            # y:(batch, Nr, F)
        Y = Y.permute(0, 2, 1).contiguous().view(-1, self.Nr)
        Y = (torch.view_as_real(Y).contiguous()).view((B, -1, 2*self.Nr))

        H = H.unsqueeze(dim=1).repeat(1,F,1,1)
        H = (torch.view_as_real(H).contiguous()).view((B, F, 2*self.Nr*2))
        HY = torch.cat((H,Y), dim=-1)          # (B, F, 2Nr+2NrNt)
        residual = self.res(HY)                # (B, F, 2Nt)
        residual = torch.view_as_complex(residual.view(B,F,2,2))

        return residual.permute(0, 2, 1)       # (batch, Nt, F)

    def forward(self, img, is_train):
        # img: (B,3,H,W); flag: training or not
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            P1 = self.P1 + self.P1_rng*(2*torch.rand(1)-1).to(device)
            P2 = self.P2 + self.P2_rng*(2*torch.rand(1)-1).to(device)
        else:
            P1 = self.P1 + self.P1_rng*torch.tensor([0]).to(device)
            P2 = self.P2 + self.P2_rng*torch.tensor([0]).to(device)

        snr_comb = torch.cat((10**(P1/10), 10**(P2/10))).unsqueeze(0)                     # [1,2]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape
        F = int(C*H*W/(2*2))                                # Nt = 2

        sig_s = torch.view_as_complex(x.view(B,-1,2))
        if self.args.distribute:
            sig_s1, sig_s2 = sig_s[:, 0:int(F/2)], sig_s[:, int(F/2):]
            sig_s1, sig_s2 = pwr_normalize(sig_s1)*10**(P1/20), pwr_normalize(sig_s2)*10**(P2/20)
            sig = torch.cat((sig_s1, sig_s2), dim = -1)
        else:
            P = (P1 + P2)/2
            sig = pwr_normalize(sig_s)*10**(P/20)               # (B, F*Nt)
        sig = sig.view(B, 2, F) 

        # assume quasi-static channel
        coef_shape = [B, self.Nr, 2]
        noise_shape = [B, self.Nr, F]

        noise = complex_sig(noise_shape, device)

        self.H = complex_sig(coef_shape, device)
        y = torch.bmm(self.H, sig) + noise                 # (B, Nr, F)

        ##### Receiver

        # calculate Power matrix
        if self.args.distribute:
            P_mat = torch.diag(torch.tensor([10**(P1/10), 10**(P2/10)]).to(torch.complex64))
        else:
            P_mat = torch.diag(torch.tensor([10**(P/10), 10**(P/10)]).to(torch.complex64))
        x_est = self.MMSE_Equ(y, self.H, P_mat)          # (B, 2, F)
        
        if self.args.res:
            x_res = self.RES_Equ(y, self.H)
            x_est = (x_est + x_res).view(B,-1)
        
        output = self.dec(torch.view_as_real(x_est).view(B,C,H,W), snr_comb)

        return output

class Div_model(nn.Module):
    def __init__(self,  args, enc, dec):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.P1 = args.P1
        self.P2 = args.P2
        self.P1_rng = args.P1_rng
        self.P2_rng = args.P2_rng

        self.Nr = args.Nr
        self.args = args

        self.enc = enc                      # Source encoder
        self.dec = dec                      # Source decoder

    def my_split(self, sig):
        # split the tensor (B, L) to (B, L/2, 2)
        B = sig.shape[0]
        sig_resp = sig.view(B, -1, 2).permute(0, 2, 1)
        sig_s1 = sig_resp[:,0:1,:]
        sig_s2 = sig_resp[:,1:,:]
        return sig_s1, sig_s2

    def my_group(self, sig1, sig2):
        # group the tensors (B, L/2) to (B, L)
        B = sig1.shape[0]
        sig = torch.cat((sig1, sig2),dim=1)
        sig = torch.permute(sig, (0,2,1)).contiguous().view(B,-1)         # (B, L)
        return sig

    def forward(self, img, is_train):
        # img: (B,3,H,W); flag: training or not
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            P1 = self.P1 + self.P1_rng*(2*torch.rand(1)-1).to(device)
            P2 = self.P2 + self.P2_rng*(2*torch.rand(1)-1).to(device)

        else:
            P1 = self.P1 + self.P1_rng*torch.tensor([0]).to(device)
            P2 = self.P2 + self.P2_rng*torch.tensor([0]).to(device)

        snr_comb = torch.cat((P1, P2)).unsqueeze(0)                     # [1,3]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape
        F = int(C*H*W/2)

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)

        # The alamouti scheme
        sig_s1, sig_s2 = self.my_split(sig_s)
        sig_A = self.my_group(sig_s1, -torch.conj(sig_s2))          # (B, L) -- [s1, -s2, s3, -s4....]
        sig_B = self.my_group(sig_s2, torch.conj(sig_s1))           # (B, L) -- [s2, s1, s4, s3....]

        sig_A, sig_B = sig_A*10**(P1/20), sig_B*10**(P2/20)
        sig = torch.cat((sig_A.view(B,1,-1), sig_B.view(B,1,-1)), dim=1)    # (B, 2, F)
        
        # assume quasi-static channel
        coef_shape = [B, self.Nr, 2]
        noise_shape = [B, self.Nr, F]

        noise = complex_sig(noise_shape, device)

        self.H = complex_sig(coef_shape, device)
        h1, h2 = self.H[:,:,0], self.H[:,:,1]
        self.Ext_H = torch.cat((torch.cat((h1, torch.conj(h2)), dim=-1).unsqueeze(-1),\
                            torch.cat((h2, -torch.conj(h1)), dim=-1).unsqueeze(-1)), dim = -1)       # (B, 2Nr, 2)
        y = torch.bmm(self.H, sig) + noise         # (B, Nr, F)

        ##### Receiver
        r1, r2 = self.my_split(y.view(B*self.Nr,F))                  
        r1, r2 = r1.view(B, self.Nr, -1), r2.view(B, self.Nr, -1)    # (B, Nr, F/2)
        z = torch.bmm(self.Ext_H.conj().permute(0,2,1), torch.cat((r1, r2.conj()), dim=1))  # (B, 2, F/2)
        z1, z2 = z[:,0,:], z[:,1,:]

        alpha1 = (torch.sum(torch.abs(h1)**2 + torch.abs(h2)**2, dim = -1) + 10**(-P1/10)).unsqueeze(-1)
        alpha2 = (torch.sum(torch.abs(h1)**2 + torch.abs(h2)**2, dim = -1) + 10**(-P2/10)).unsqueeze(-1)
        s1, s2 = z1/alpha1, z2/alpha2                                       # (B, F/2)
        x_est = self.my_group(s1.view(B,1,-1), s2.view(B,1,-1))           # (B, F)
        
        y_comb = torch.view_as_real(x_est).view(B,C,H,W)

        output = self.dec(y_comb, snr_comb)

        return output


class Div_model3(nn.Module):
    def __init__(self,  args, enc, dec):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.P = args.P1
        self.P_rng = args.P1_rng


        self.Nr = args.Nr
        self.args = args

        self.enc = enc                      # Source encoder
        self.dec = dec                      # Source decoder

    def effective_channel(self, H):
        # effective channel gain
        channel_gain = 2*torch.sum(torch.abs(H)**2, dim = -1)
        # effctive H
        h1, h2, h3 = H[:,:,0], H[:,:,1], H[:,:,2]    # (B, Nr)
        h0 = torch.zeros_like(h1).to(h1.device)
        # total 4 rows
        row1 = torch.cat((h1, h2, h3, h0, torch.conj(h1), torch.conj(h2), torch.conj(h3), h0)).unsqueeze(dim=1)
        row2 = torch.cat((h2, -h1, h0, h3, torch.conj(h2), -torch.conj(h1), h0, torch.conj(h3))).unsqueeze(dim=1)
        row3 = torch.cat((h3, h0, -h1, -h2, torch.conj(h3), h0, -torch.conj(h1), -torch.conj(h2))).unsqueeze(dim=1)
        row4 = torch.cat((h0, -h3, h2, -h1, h0, -torch.conj(h3), torch.conj(h2), -torch.conj(h1))).unsqueeze(dim=1)
        H_eff = torch.conj(torch.cat((row1, row2, row3, row4),dim=1))    # (B, 4, 8Nr)

        return channel_gain, H_eff

    def forward(self, img, is_train):
        # img: (B,3,H,W); flag: training or not
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            P = self.P + self.P_rng*(2*torch.rand(1)-1).to(device)
        else:
            P = self.P + self.P_rng*torch.tensor([0]).to(device)

        snr_comb = P.unsqueeze(0)                     # [1,3]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape
        F = int(C*H*W/2)

        sig_s = x.view(B,-1,2)
        sig_s = torch.view_as_complex(sig_s)
        sig_s = pwr_normalize(sig_s)*10**(P/20)


        # assume quasi-static channel
        coef_shape = [B, self.Nr, 3]
        noise_shape = [B, self.Nr*8, F]

        noise = complex_sig(noise_shape, device)

        H = complex_sig(coef_shape, device)
        
        ## for simplification, we only consider the effective channel
        channel_gain, H_eff = self.effective_channel(H)
        noise_eff = torch.bmm(H_eff, noise)     # (B, 4, F)
        noise_eff = noise_eff.permute(0,2,1).view(B,-1)

        y_eff = channel_gain*sig_s + noise_eff
        # MMSE equalization
        x_est= y_eff/(channel_gain + 10**(-P/10)) # (B,-1)

        y_comb = torch.view_as_real(x_est).view(B,C,H,W)

        output = self.dec(y_comb, snr_comb)

        return output

class Mul_model3(nn.Module):
    def __init__(self, args, enc, dec, res = None):
        super().__init__()

        self.c_feat = args.cfeat
        self.c_out = args.cout

        self.P = args.P1
        self.P_rng = args.P1_rng
        self.Nr = args.Nr
        self.Nt = args.Nt
        self.args = args

        self.enc = enc                      # Enc User B
        self.dec = dec                      # Source decoder
        self.res = res                      # Residual

    def MMSE_Equ(self, y, H, P):
        # MMSE Equ
        B = y.shape[0]
        P = P.unsqueeze(0).repeat(B,1,1).to(y.device)  # (B, 3, 3)
        HP = torch.bmm(P, H.permute(0,2,1).conj())
        RHH = torch.bmm(H, HP)                       # (B, Nr, Nr)
        eye = torch.eye(self.Nr, dtype=torch.cfloat).to(y.device).repeat(B, 1, 1)
        inv_Ryy = torch.inverse(RHH + eye)
        G = torch.bmm(HP, inv_Ryy)                   # (B, 3, Nr)
        x_equ = torch.bmm(G, y)
        return x_equ

    def RES_Equ(self, Y, H):
        # Residual Equ
        B, F = Y.shape[0], Y.shape[2]                            # y:(batch, Nr, F)
        Y = Y.permute(0, 2, 1).contiguous().view(-1, self.Nr)
        Y = (torch.view_as_real(Y).contiguous()).view((B, -1, 2*self.Nr))

        H = H.unsqueeze(dim=1).repeat(1,F,1,1)
        H = (torch.view_as_real(H).contiguous()).view((B, F, 2*self.Nr*self.Nt))
        HY = torch.cat((H,Y), dim=-1)          # (B, F, 2Nr+2NrNt)
        residual = self.res(HY)                # (B, F, 2Nt)
        residual = torch.view_as_complex(residual.view(B,F,self.Nt,2))

        return residual.permute(0, 2, 1)       # (batch, Nt, F)

    def forward(self, img, is_train):
        # img: (B,3,H,W); flag: training or not
        device = img.device

        # channel snr settings
        if self.args.adapt and is_train:
            P = self.P + self.P_rng*(2*torch.rand(1)-1).to(device)
        else:
            P = self.P + self.P_rng*torch.tensor([0]).to(device)

        snr_comb = 10**(P/10).unsqueeze(0)                     # [1,2]

        # Source node
        x = self.enc(img, snr_comb)
        B,C,H,W = x.shape
        F = int(C*H*W/(2*self.Nt))                                # Nt = 2

        sig_s = torch.view_as_complex(x.view(B,-1,2))
        sig = pwr_normalize(sig_s)*10**(P/20)               # (B, F*Nt)
        sig = sig.view(B, self.Nt, F) 

        # assume quasi-static channel
        coef_shape = [B, self.Nr, self.Nt]
        noise_shape = [B, self.Nr, F]

        noise = complex_sig(noise_shape, device)

        self.H = complex_sig(coef_shape, device)
        y = torch.bmm(self.H, sig) + noise                 # (B, Nr, F)

        ##### Receiver

        # calculate Power matrix
        P_mat = 10**(P/10)*torch.eye(self.Nt, dtype=torch.cfloat).to(y.device)
        x_est = self.MMSE_Equ(y, self.H, P_mat)          # (B, 3, F)
        
        if self.args.res:
            x_res = self.RES_Equ(y, self.H)
            x_est = (x_est + x_res).view(B,-1)
        
        output = self.dec(torch.view_as_real(x_est).view(B,C,H,W), snr_comb)

        return output