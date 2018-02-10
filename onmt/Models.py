from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import onmt
from onmt.Utils import aeq


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Context]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """
    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            hidden (class specific):
               initial hidden state.

        Returns:k
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                   `[layers x batch x hidden]`
                * contexts for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        return (mean, mean), emb


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.no_pack_padded_seq = False

        # Use pytorch version when available.
        if rnn_type == "SRU":
            # SRU doesn't support PackedSequence.
            self.no_pack_padded_seq = True
            self.rnn = onmt.modules.SRU(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)
        else:
            self.rnn = getattr(nn, rnn_type)(
                    input_size=embeddings.embedding_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional)

    def forward(self, input, lengths=None, hidden=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class LuaBiRNNEncoder(EncoderBase):
    """
    Bi-directional RNN lua like RNN encoder:
    train two multilayer uni-directional RNN with input seq and
    inversed seq, then concat them afterwards
    Gated: extract answer from final output and
           gate for the final output of the encoder but not consider the final hidden state
    """
    def __init__(self, rnn_type, num_layers, hidden_size,
                 dropout, embeddings):
        super(LuaBiRNNEncoder, self).__init__()

        assert rnn_type == "LSTM"
        hidden_size = hidden_size // 2
        self.embeddings = embeddings
        self.no_pack_padded_seq = False
        self.bidirectional = False

        self.rnn_fw = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
                input_size=embeddings.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=self.bidirectional)
        self.rnn_bw = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
                input_size=embeddings.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=self.bidirectional)
        #self.fc1 = nn.Linear(4*hidden_size, 2*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, 2*hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, lengths=None, hidden=None):
        """train fw and bw separately and then concatenate"""
        self._check_args(input, lengths, hidden)

        def reverse(seq_data_input, lengths):
            """Reverse Function for backward RNN input and output"""
            seq_data = seq_data_input.clone()
            data_reversed = []
            for i, l in enumerate(lengths):
                # reverse the real data part and remain the paded part unchanged
                idx_reversed = [j for j in range(l-1, -1, -1)] + [k for k in range(l, seq_data.size(0))]
                assert len(idx_reversed) == seq_data.size(0)
                # while len(idx_reversed) < seq_data.size(0):
                #     idx_reversed.append(1) # PAD index may change
                idx_reversed = Variable(torch.LongTensor(idx_reversed), requires_grad=False).cuda()
                input_reversed_i = seq_data[:, i, :].index_select(0, idx_reversed)
                input_reversed_i = torch.unsqueeze(input_reversed_i, dim=1)
                data_reversed.append(input_reversed_i)
            data_reversed = torch.cat(data_reversed, 1)
            return data_reversed

        def get_ans_posi(is_ans_feature):
            is_ans = is_ans_feature.clone() # [seq_len, batch_size]
            # Bad code, need know the idx of answer indicator
            is_ans = is_ans == 3
            batch_size = is_ans.size()[1]
            ans_posi = torch.stack([torch.cat((is_ans[:, i].nonzero()[0], is_ans[:, i].nonzero()[-1] + 1))
                                    for i in range(batch_size)], dim=1)
            return ans_posi.data

        emb = self.embeddings(input)
        lengths = lengths.view(-1).tolist()
        input_reversed = reverse(input, lengths)
        emb_reversed = self.embeddings(input_reversed)
        s_len, batch, emb_dim = emb.size()
        s_len_reversed, batch_reversed, emb_dim_reversed = emb_reversed.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(emb, lengths)
        packed_emb_reversed = emb_reversed
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_emb_reversed = pack(emb_reversed, lengths)

        # forward
        outputs_fw, hidden_t_fw = self.rnn_fw(packed_emb, hidden)
        # backward
        outputs_bw, hidden_t_bw = self.rnn_bw(packed_emb_reversed, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            # forward output
            outputs_fw = unpack(outputs_fw)[0]
            # backward output
            outputs_bw = unpack(outputs_bw)[0]
            outputs_bw = reverse(outputs_bw, lengths)

        outputs = torch.cat((outputs_fw, outputs_bw), -1)   # [seq_len, batch_size, 2*hidden]

        # doing selective encoding here
        my_ans_posi = get_ans_posi(input[:, :, -1])  # [2, batch_size]
        # s1
        ans_BiFibalState = torch.stack([torch.cat((outputs_fw[my_ans_posi[1, i] - 1, i, :],
                                                   outputs_bw[my_ans_posi[0, i], i, :])) for i in range(batch)],
                                       dim=0)   # [batch_size, 2*hidden]

        # s2
        # ans_BiFibalState = torch.stack([torch.cat((outputs[my_ans_posi[1, i] - 1, i, :],
        #                                           outputs[my_ans_posi[0, i], i, :])) for i in range(batch)],
        #                               dim=0)   # [batch_size, 4*hidden]
        # ans_BiFibalState = self.sigmoid(self.fc1(ans_BiFibalState))   # [batch_size, 2*hidden]
        #
        # ans_BiFibalState = ans_BiFibalState.unsqueeze(0).expand(outputs.size()) # [seq_len, batch_size, 2*hidden]

        gate_fc = self.fc2(torch.cat([outputs, ans_BiFibalState], dim=2))    # [seq_len, batch_size, 2*hidden]
        # gate_fc2 = self.fc2(ans_BiFibalState)   # [batch_size, 2*hidden]
        gate = self.sigmoid(gate_fc)    # [seq_len, batch_size, 2*hidden]
        outputs = outputs * gate

        hidden_t = []
        hidden_t.append(torch.stack((hidden_t_fw[0][0,:,:], hidden_t_bw[0][0,:,:],
                                   hidden_t_fw[0][1, :, :],hidden_t_bw[0][1,:,:]), 0))
        hidden_t.append(torch.stack((hidden_t_fw[1][0, :, :], hidden_t_bw[1][0, :, :],
                                   hidden_t_fw[1][1, :, :], hidden_t_bw[1][1, :, :]), 0))
        hidden_t = tuple(hidden_t)

        return hidden_t, outputs


class PyBiRNNEncoder(EncoderBase):
    """
    Bi-directional RNN pytroch like RNN encoder.
    No Gate.
    """
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout, embeddings):
        super(PyBiRNNEncoder, self).__init__()

        assert rnn_type == "LSTM"
        assert bidirectional
        assert num_layers == 2

        hidden_size = hidden_size // 2
        self.real_hidden_size = hidden_size
        self.no_pack_padded_seq = False
        self.bidirectional = bidirectional

        self.embeddings = embeddings
        self.bi_rnn1 = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
                input_size=embeddings.embedding_size,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=self.bidirectional)
        self.bi_rnn2 = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
                input_size=2 * hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, lengths=None, hidden=None):
        """Embedding->BiRNN1->Dropout->BiRNN2"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        lengths = lengths.view(-1).tolist()
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(emb, lengths)
        # forward
        outputs1, hidden_t1 = self.bi_rnn1(packed_emb, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            # output1
            outputs1 = unpack(outputs1)[0]

        outputs1 = self.dropout(outputs1)

        packed_input2 = outputs1
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_input2 = pack(outputs1, lengths)
        # forward
        outputs2, hidden_t2 = self.bi_rnn2(packed_input2, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            # output1
            outputs2 = unpack(outputs2)[0]

        hidden_t = []
        # h
        hidden_t.append(torch.cat([hidden_t1[0], hidden_t2[0]], dim=0))
        # c
        hidden_t.append(torch.cat([hidden_t1[1], hidden_t2[1]], dim=0))
        hidden_t = tuple(hidden_t)

        return hidden_t, outputs2


class PyBiRNNEncoder2(EncoderBase):
    """
    Bi-directional RNN pytorch like RNN encoder.
    Extract the answer output and Gating for both BiRNN-layer's output
    Also gate for the final state
    """

    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout, embeddings):
        super(PyBiRNNEncoder2, self).__init__()

        assert rnn_type == "LSTM"
        assert bidirectional
        assert num_layers == 2

        hidden_size = hidden_size // 2
        self.real_hidden_size = hidden_size
        self.no_pack_padded_seq = False
        self.bidirectional = bidirectional

        self.embeddings = embeddings
        self.bi_rnn1 = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=embeddings.embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional)
        self.bi_rnn2 = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, lengths=None, hidden=None):
        """
        Extract the answer output and Gating for both BiRNN-layer's output
        Also gate for the final state
        """
        self._check_args(input, lengths, hidden)

        def get_ans_posi(is_ans_feature):
            is_ans = is_ans_feature.clone()  # [seq_len, batch_size]
            # Bad code, need know the idx of answer indicator
            is_ans = is_ans == 3
            batch_size = is_ans.size()[1]
            ans_posi = torch.stack([torch.cat((is_ans[:, i].nonzero()[0], is_ans[:, i].nonzero()[-1] + 1))
                                    for i in range(batch_size)], dim=1)
            return ans_posi.data

        def get_gates(outputs, state, my_ans_posi, layer):
            h_state = state[0].clone()
            c_state = state[1].clone()
            # s1
            # ans_BiFibalState = torch.stack([0.5 * outputs[my_ans_posi[1, i] - 1, i, :] +
            #                                 0.5 * outputs[my_ans_posi[0, i], i, :] for i in range(batch)],
            #                              dim=0)  # [batch_size, 2*hidden]

            # s4
            ans_BiFibalState = torch.stack([torch.mean(outputs[my_ans_posi[0, i]:my_ans_posi[1, i], i, :], dim=0)
                                            for i in range(batch)], dim=0)  # [batch_size, 2*hidden]
            out_BiFibalState = ans_BiFibalState.unsqueeze(0).expand(outputs.size())
            out_gate_fc = layer(torch.cat([outputs, out_BiFibalState], dim=2))  # [seq_len, batch_size, 2*hidden]
            out_gate = self.sigmoid(out_gate_fc)  # [seq_len, batch_size, 2*hidden]

            h_state = torch.cat([h_state[0], h_state[1]], dim=-1)  # [batch_size, 2*hidden]
            h_state_gate_fc = layer(
                torch.cat([h_state, ans_BiFibalState], dim=1))  # [batch_size, 4*hidden] -> [batch_size, 2*hidden]
            h_state_gate = self.sigmoid(h_state_gate_fc)  # [batch_size, 2*hidden]
            h_state_gate = torch.stack(
                [h_state_gate[:, : self.real_hidden_size], h_state_gate[:, self.real_hidden_size:]],
                dim=0)  # [2, batch_size, hidden]

            c_state = torch.cat([c_state[0], c_state[1]], dim=-1)
            c_state_gate_fc = layer(
                torch.cat([c_state, ans_BiFibalState], dim=1))  # [batch_size, 4*hidden] -> [batch_size, 2*hidden]
            c_state_gate = self.sigmoid(c_state_gate_fc)  # [batch_size, 2*hidden]
            c_state_gate = torch.stack(
                [c_state_gate[:, : self.real_hidden_size], c_state_gate[:, self.real_hidden_size:]],
                dim=0)  # [2, batch_size, hidden]

            return out_gate, h_state_gate, c_state_gate

        emb = self.embeddings(input)
        lengths = lengths.view(-1).tolist()
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(emb, lengths)
        # forward
        outputs1, hidden_t1 = self.bi_rnn1(packed_emb, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            # output1
            outputs1 = unpack(outputs1)[0]

        # doing selective encoding here
        outputs1 = self.dropout(outputs1)
        my_ans_posi = get_ans_posi(input[:, :, -1])  # [2, batch_size]
        out_gate1, h_state_gate1, c_state_gate1 = get_gates(outputs1, hidden_t1, my_ans_posi, self.fc1)
        outputs1 = outputs1 * out_gate1
        hidden_t1 = list(hidden_t1)
        hidden_t1[0] = hidden_t1[0] * h_state_gate1
        hidden_t1[1] = hidden_t1[1] * c_state_gate1
        hidden_t1 = tuple(hidden_t1)

        packed_input2 = outputs1
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_input2 = pack(outputs1, lengths)
        # forward
        outputs2, hidden_t2 = self.bi_rnn2(packed_input2, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            # output1
            outputs2 = unpack(outputs2)[0]

        # doing selective encoding here
        out_gate2, h_state_gate2, c_state_gate2 = get_gates(outputs2, hidden_t2, my_ans_posi, self.fc2)
        outputs2 = outputs2 * out_gate2
        hidden_t2 = list(hidden_t2)
        hidden_t2[0] = hidden_t2[0] * h_state_gate2
        hidden_t2[1] = hidden_t2[1] * c_state_gate2
        hidden_t2 = tuple(hidden_t2)

        hidden_t = []
        hidden_t.append(torch.cat([hidden_t1[0], hidden_t2[0]], dim=0))
        hidden_t.append(torch.cat([hidden_t1[1], hidden_t2[1]], dim=0))
        hidden_t = tuple(hidden_t)

        return hidden_t, outputs2


class PyBiRNNEncoder3(EncoderBase):
    """
    Bi-directional RNN, Reading Comprehension like RNN encoder
    (src->BiRNN1, ans->BiRNN3->average)->gated output -> BiRNN2->Decoder
    """

    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout, embeddings):
        super(PyBiRNNEncoder3, self).__init__()

        assert rnn_type == "LSTM"
        assert bidirectional
        assert num_layers == 2

        hidden_size = hidden_size // 2
        self.real_hidden_size = hidden_size
        self.no_pack_padded_seq = False
        self.bidirectional = bidirectional

        self.embeddings = embeddings
        self.bi_rnn1 = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=embeddings.embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional)
        self.bi_rnn2 = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional)
        self.ans_bi_rnn = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=embeddings.embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional)
        self.fc1 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        # self.fc2 = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input, lengths=None, hidden=None):
        """
        a reading comprehension style framework for encoder
        (src->BiRNN1, ans->BiRNN3->average)->gated output -> BiRNN2->Decoder
        """
        assert isinstance(input, tuple)
        assert isinstance(lengths, tuple)
        src_input, ans_input = input
        # Bad code, need to know the vocab idx of <blank>
        src_mask = (src_input[:, :, 0] != 1).float()
        ans_mask = (ans_input[:, :, 0] != 1).float()
        extend_ans_mask = ans_mask.unsqueeze(2)
        src_lengths, ans_lengths = lengths
        self._check_args(src_input, src_lengths, hidden)
        self._check_args(ans_input, ans_lengths, hidden)

        src_emb = self.embeddings(src_input)    # [src_seq_len, batch_size, emb_dim]
        src_lengths = src_lengths.view(-1).tolist()
        s_len, batch, emb_dim = src_emb.size()

        ans_emb = self.embeddings(ans_input)  # [ans_seq_len, batch_size, emb_dim]
        ans_lengths = ans_lengths.view(-1).tolist()
        ans_len, batch, emb_dim = ans_emb.size()

        packed_emb = src_emb
        if src_lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_emb = pack(src_emb, src_lengths)
        # forward
        src_outputs1, src_hidden_t1 = self.bi_rnn1(packed_emb, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            # output1
            src_outputs1 = unpack(src_outputs1)[0]

        # doing selective encoding here
        src_outputs1 = self.dropout(src_outputs1)

        # answer encoding
        # the final hidden state is incorrect because without considering various lengths of sequence
        ans_outputs, _ = self.ans_bi_rnn(ans_emb, hidden)
        ans_outputs = ans_outputs * extend_ans_mask #[ans_len, batch, 2*hidden_size]
        mean_ans_output = torch.mean(ans_outputs, dim=0)    #[batch, 2*hidden_size]
        mean_ans_output = mean_ans_output.unsqueeze(0).expand(src_outputs1.size())  #[src_len, batch, 2*hidden_size]

        # calculate the output gate for src_outputs1
        gate_src_outs1 = self.fc1(torch.cat([src_outputs1, mean_ans_output], dim=2))

        src_outputs1 *= gate_src_outs1

        packed_input2 = src_outputs1
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_input2 = pack(src_outputs1, src_lengths)
        # forward
        src_outputs2, src_hidden_t2 = self.bi_rnn2(packed_input2, hidden)
        if lengths is not None and not self.no_pack_padded_seq:
            # output2
            src_outputs2 = unpack(src_outputs2)[0]

        hidden_t = []
        hidden_t.append(torch.cat([src_hidden_t2[0], src_hidden_t2[0].clone()], dim=0))
        hidden_t.append(torch.cat([src_hidden_t2[1], src_hidden_t2[1].clone()], dim=0))
        hidden_t = tuple(hidden_t)

        return hidden_t, src_outputs2


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Context]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type, self._input_size, hidden_size,
                                   num_layers, dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type
            )
            self._copy = True

    def forward(self, input, context, state, context_lengths=None):
        """
        Args:
            input (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            context (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            context_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * outputs: output from the decoder
                         `[tgt_len x batch x hidden]`.
                * state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Args Check
        assert isinstance(state, RNNDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)
        # END Args Check

        # Run the forward pass of the RNN.
        hidden, outputs, attns, coverage = self._run_forward_pass(
            input, context, state, context_lengths=context_lengths)

        # Update the state with the result.
        final_output = outputs[-1]
        state.update_state(hidden, final_output.unsqueeze(0),
                           coverage.unsqueeze(0)
                           if coverage is not None else None)

        # Concatenates sequence of tensors along a new dimension.
        outputs = torch.stack(outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])

        return outputs, state, attns

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim.
        We need to convert it to layers x batch x (directions*dim).
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, src, context, enc_hidden):
        if isinstance(enc_hidden, tuple):  # LSTM
            return RNNDecoderState(context, self.hidden_size,
                                   tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:  # GRU
            return RNNDecoderState(context, self.hidden_size,
                                   self._fix_enc_hidden(enc_hidden))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """
    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            input (LongTensor): a sequence of input tokens tensors
                                of size (len x batch x nfeats).
            context (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            context_lengths (LongTensor): the source context lengths.
        Returns:
            hidden (Variable): final hidden state from the decoder.
            outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
            coverage (FloatTensor, optional): coverage from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        coverage = None

        emb = self.embeddings(input)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, hidden = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, hidden = self.rnn(emb, state.hidden)
        # Result Check
        input_len, input_batch, _ = input.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(input_len, output_len)
        aeq(input_batch, output_batch)
        # END Result Check

        # Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            rnn_output.transpose(0, 1).contiguous(),  # (output_len, batch, d)
            context.transpose(0, 1),                  # (contxt_len, batch, d)
            context_lengths=context_lengths
        )
        attns["std"] = attn_scores

        # Calculate the context gate.
        if self.context_gate is not None:
            outputs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                attn_outputs.view(-1, attn_outputs.size(2))
            )
            outputs = outputs.view(input_len, input_batch, self.hidden_size)
            outputs = self.dropout(outputs)
        else:
            outputs = self.dropout(attn_outputs)    # (input_len, batch, d)

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        """
        Private helper for building standard decoder RNN.
        """
        # Use pytorch version when available.
        if rnn_type == "SRU":
            return onmt.modules.SRU(
                    input_size, hidden_size,
                    num_layers=num_layers,
                    dropout=dropout)

        return getattr(nn, rnn_type)(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout)

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Context n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, input, context, state, context_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        output = state.input_feed.squeeze(0)
        output_batch, _ = output.size()
        input_len, input_batch, _ = input.size()
        aeq(input_batch, output_batch)
        # END Additional args check.

        # Initialize local and return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        hidden = state.hidden
        coverage = state.coverage.squeeze(0) \
            if state.coverage is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            emb_t = torch.cat([emb_t, output], 1)

            rnn_output, hidden = self.rnn(emb_t, hidden)
            attn_output, attn = self.attn(
                rnn_output,
                context.transpose(0, 1),
                context_lengths=context_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                output = self.context_gate(
                    emb_t, rnn_output, attn_output
                )
                output = self.dropout(output)
            else:
                output = self.dropout(attn_output)
            outputs += [output]
            attns["std"] += [attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + attn \
                    if coverage is not None else attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy:
                _, copy_attn = self.copy_attn(output,
                                              context.transpose(0, 1))
                attns["copy"] += [copy_attn]

        # Return result.
        return hidden, outputs, attns, coverage

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class SelEncNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(SelEncNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, ans, tgt, src_lengths, ans_lengths, dec_state=None):
        """Forward propagate a `src`, `ans` and `tgt` triple for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            ans (:obj:`Tensor`):
                an answer sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. Only for `Text`
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            src_lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            ans_lengths(:obj:`LongTensor`): the ans lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_hidden, context = self.encoder((src, ans), (src_lengths, ans_lengths))
        enc_state = self.decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, context,
                                             enc_state if dec_state is None
                                             else dec_state,
                                             context_lengths=src_lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnnstate):
        """
        Args:
            context (FloatTensor): output from the encoder of size
                                   len x batch x rnn_size.
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate (Variable): final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
            input_feed (FloatTensor): output from last layer of the decoder.
            coverage (FloatTensor): coverage output from the decoder.
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.coverage = None

        # Init the input feed.
        batch_size = context.size(1)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
